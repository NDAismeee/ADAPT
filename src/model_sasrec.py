import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    """
    FFN used in SASRec:
    Linear -> GELU/ReLU -> Dropout -> Linear -> Dropout (with residual outside)
    """
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Custom multi-head self-attention with masking.
    Inputs: x (B, L, H)
    Mask: attn_mask (B, 1, L, L) where True = allowed, False = blocked
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product: (B, heads, L, L)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 1) mask bằng số âm lớn thay vì -inf
        scores = scores.masked_fill(~attn_mask, -1e9)

        # 2) phát hiện query rows bị mask hết key
        row_has_any_key = attn_mask.any(dim=-1, keepdim=True)  # (B,1,L,1)

        # 3) với những rows không có key nào hợp lệ -> set scores = 0 để softmax không NaN
        scores = scores.masked_fill(~row_has_any_key, 0.0)

        attn = torch.softmax(scores, dim=-1)

        # 4) cuối cùng zero-out attention ở những query rows đó
        attn = attn * row_has_any_key
        # =======================================================

        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, H)  # (B, L, H)

        out = self.out_proj(out)
        out = self.proj_dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = PointWiseFeedForward(hidden_size, dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # attn_mask: (B, 1, L, L)

        # ===== Self-attention (pre-norm) =====
        h = self.ln1(x)
        h = self.attn(h, attn_mask)
        x = x + self.dropout(h)

        # >>> FIX 1: zero-out padding positions <<<
        pad_mask = attn_mask.any(dim=-1).squeeze(1).float()  # (B, L)
        x = x * pad_mask.unsqueeze(-1)
        # ======================================

        # ===== FFN (pre-norm) =====
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        # >>> FIX 2: zero-out padding positions (again) <<<
        x = x * pad_mask.unsqueeze(-1)
        # =================================================

        return x


class SASRec(nn.Module):
    """
    SASRec model.
    - input_seq: (B, L) with 0 as padding (left padded in your datasets.py)
    Train forward returns pos_logits, neg_logits: (B, L)
    Predict returns logits for candidates: (B, C)
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # item_id: 0..num_items ; 0 reserved for padding
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size, eps=1e-12)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        # padding embedding stays near 0
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    def _build_attn_mask(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Build attention mask that combines:
        - padding mask (do not attend to padded positions)
        - causal mask (no future leakage)
        Return: (B, 1, L, L) boolean mask, True = allowed
        """
        device = input_seq.device
        B, L = input_seq.shape

        # padding mask: True where real tokens
        pad_mask = (input_seq != 0)  # (B, L)

        # causal mask: lower triangular (L, L)
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

        # combine:
        # key positions must be real tokens; causal enforces i attends to <= i
        # (B, 1, L, L)
        attn_mask = causal.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        key_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
        attn_mask = attn_mask & key_mask

        return attn_mask

    def _encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence into hidden states.
        Returns: (B, L, H)
        """
        B, L = input_seq.shape
        device = input_seq.device

        # positions: 0..L-1
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_seq) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        x = self.ln(x)

        attn_mask = self._build_attn_mask(input_seq)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        return x

    def forward(
        self,
        input_seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Train forward:
          input_seq: (B, L)
          pos_items: (B, L)
          neg_items: (B, L)
        Returns:
          pos_logits: (B, L)
          neg_logits: (B, L)
        """
        h = self._encode(input_seq)  # (B, L, H)

        pos_emb = self.item_emb(pos_items)  # (B, L, H)
        neg_emb = self.item_emb(neg_items)  # (B, L, H)

        pos_logits = (h * pos_emb).sum(dim=-1)  # (B, L)
        neg_logits = (h * neg_emb).sum(dim=-1)  # (B, L)
        return pos_logits, neg_logits

    @torch.no_grad()
    def predict(self, input_seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Eval predict:
          input_seq: (B, L)
          candidates: (B, C) where candidates[0] is positive, rest are negatives
        Returns:
          logits: (B, C)
        """
        h = self._encode(input_seq)  # (B, L, H)

        # Since your dataset left-pads, the last position is the most recent step
        h_last = h[:, -1, :]  # (B, H)

        cand_emb = self.item_emb(candidates)  # (B, C, H)
        logits = torch.bmm(cand_emb, h_last.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return logits


def sasrec_bce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    pos_items: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Standard SASRec loss with negative sampling:
      loss = -log(sigmoid(pos)) - log(1 - sigmoid(neg))
    Ignore padded positions where pos_items == 0.
    """
    mask = (pos_items != 0).float()

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="none"
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="none"
    )

    loss = (pos_loss + neg_loss) * mask

    if reduction == "sum":
        return loss.sum()
    # default mean over non-pad tokens
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom
