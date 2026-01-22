import copy
from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from model_sasrec import SASRec, sasrec_bce_loss
from fl_datasets import UserSequenceDataset
from fl_utils import clone_state_dict, compute_cosine_similarity
from openai_agents import OpenAIGeneratorAgent

from verify import TrainingVerifier


# =========================================================
# Helper: get model top-K predictions for one user
# =========================================================
@torch.no_grad()
def get_model_topk(model, user_seq, num_items, k, device):
    model.eval()
    
    # Model SASRec khởi tạo Positional Embedding có kích thước cố định (ví dụ 50).
    # Nếu truyền vào sequence dài 100, model sẽ crash khi truy cập index 51.
    max_len = getattr(model, 'max_seq_len', 50) 
    if len(user_seq) > max_len:
        user_seq = user_seq[-max_len:]

    seq = torch.tensor(user_seq, dtype=torch.long, device=device).unsqueeze(0)
    
    # Item ID phải nằm trong khoảng [0, num_items].
    limit = getattr(model.item_emb, 'num_embeddings', num_items + 1)
    if seq.max() >= limit:
         # Clamp về item hợp lệ cuối cùng (hoặc 0) để tránh sập GPU
         seq = torch.clamp(seq, 0, limit - 1)

    pos_items = torch.zeros_like(seq)
    neg_items = torch.zeros_like(seq)

    logits, _ = model(seq, pos_items, neg_items)

    if logits is None:
        return []

    # logits expected: [B, T, V] or [V]
    if logits.dim() == 3:
        scores = logits[0, -1]        # [V] Lấy bước thời gian cuối cùng
    elif logits.dim() == 2:
        scores = logits[0]            # [V]
    elif logits.dim() == 1:
        scores = logits               # [V]
    else:
        return []

    if scores.dim() != 1 or scores.numel() == 0:
        return []

    max_k = scores.numel()
    k = min(int(k), int(max_k))
    if k <= 0:
        return []

    topk = torch.topk(scores, k=k).indices.tolist()
    return [int(i) for i in topk]


# =========================================================
# Helper: one lightweight training step
# =========================================================
def local_one_step(model, dataloader, device, lr: float = 1e-3):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)

    for batch in dataloader:
        input_seq = batch["input_seq"].to(device)
        pos_item = batch["pos_item"].to(device)
        neg_item = batch["neg_item"].to(device)

        pos_items = pos_item.expand(-1, input_seq.size(1))
        neg_items = neg_item.expand(-1, input_seq.size(1))

        optimizer.zero_grad()
        pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
        loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
        loss.backward()
        optimizer.step()
        break  # exactly ONE step


@torch.no_grad()
def eval_val_loss(model, dataloader, device, max_batches: int = 5) -> float:
    """
    Lightweight validation loss estimation
    """
    model.eval()
    losses = []
    n = 0

    for batch in dataloader:
        input_seq = batch["input_seq"].to(device)
        pos_item = batch["pos_item"].to(device)
        neg_item = batch["neg_item"].to(device)

        pos_items = pos_item.expand(-1, input_seq.size(1))
        neg_items = neg_item.expand(-1, input_seq.size(1))

        pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
        loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
        losses.append(loss.item())

        n += 1
        if n >= max(1, max_batches):
            break

    return float(sum(losses) / max(len(losses), 1))


# =========================================================
# Helper: compute diversity
# =========================================================
def compute_diversity_ratio(user_seq: list) -> float:
    if not user_seq:
        return 0.0
    return len(set(user_seq)) / len(user_seq)


# =========================================================
# Helper: compute update norm
# =========================================================
@torch.no_grad()
def compute_update_norm(local_state: Dict[str, torch.Tensor], global_state: Dict[str, torch.Tensor]) -> float:
    sq = 0.0
    for k, v in local_state.items():
        if k not in global_state:
            continue
        dv = (v.detach().float().cpu() - global_state[k].detach().float().cpu())
        sq += float(torch.sum(dv * dv).item())
    return float(sq ** 0.5)


# =========================================================
# Helper: build signal skeleton
# =========================================================
def _init_signal() -> Dict[str, Any]:
    return {
        "performance": {},
        "update_quality": {},
        "distribution_shift": {},
        "adapt_meta": {
            "accepted": False,
            "n_synth": 0,
            "regime": "none"
        },
    }


# =========================================================
# Helper: fill common signals
# =========================================================
def _fill_common_signals(
    client_signal: Dict[str, Any],
    original_seq_len: int,
    final_seq_len: int,
    last_loss: Optional[float],
    config: dict,
    user_seq: list
):
    # Performance
    client_signal["performance"]["train_loss"] = float(last_loss) if last_loss is not None else None

    # Thresholds
    short_seq_th = config.get("short_seq_threshold", 20)
    high_loss_th = config.get("high_loss_threshold", 0.6)

    # Boolean flags for simple logic
    client_signal["performance"]["is_short_history"] = (int(original_seq_len) < int(short_seq_th))
    client_signal["performance"]["is_high_loss"] = (
        last_loss is not None and float(last_loss) > float(high_loss_th)
    )

    # Distribution / Diversity
    div_score = compute_diversity_ratio(user_seq)
    client_signal["distribution_shift"]["diversity_score"] = div_score
    client_signal["distribution_shift"]["seq_len"] = int(original_seq_len)
    
    # Meta stats
    client_signal["distribution_shift"]["n_synth"] = int(client_signal["adapt_meta"]["n_synth"])


# =========================================================
# Client update
# =========================================================
def client_update(
    user_id: int,
    user_seq: list,
    global_state_dict: dict,
    num_items: int,
    config: dict,
    device: torch.device,
    generation_budget: int = 0,
    probe_only: bool = False,
    global_aux_data: Dict[str, Any] = None,
) -> Tuple[dict, int, Dict[str, Any]]:
    """
    Perform local training on ONE client.
    Includes ADAPT logic: Prompt Routing -> Generation -> Verification.
    """

    # --------------------------------------------------
    # 0. Prepare signal skeleton
    # --------------------------------------------------
    client_signal: Dict[str, Any] = _init_signal()
    original_seq_len = len(user_seq)

    # --------------------------------------------------
    # 1. Skip very short users
    # --------------------------------------------------
    if len(user_seq) < 2:
        _fill_common_signals(client_signal, original_seq_len, original_seq_len, None, config, user_seq)
        return clone_state_dict(global_state_dict), 0, client_signal

    # --------------------------------------------------
    # 2. PROBE ONLY: Calculate pre-training signals
    # --------------------------------------------------
    if probe_only:
        probe_model = SASRec(
            num_items=num_items,
            max_seq_len=config["max_seq_len"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)
        probe_model.load_state_dict(global_state_dict, strict=True)

        probe_dataset = UserSequenceDataset(
            user_seq=user_seq,
            max_seq_len=config["max_seq_len"],
            num_items=num_items,
        )
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=config.get("local_batch_size", 32),
            shuffle=True,
            drop_last=False,
        )

        probe_loss: Optional[float] = None
        try:
            local_one_step(probe_model, probe_loader, device, lr=config.get("probe_lr", 1e-3))
            probe_loss = eval_val_loss(
                probe_model,
                probe_loader,
                device,
                max_batches=config.get("probe_max_val_batches", 3),
            )
        except Exception as e:
            if config.get("agent_debug", False):
                print(f"[Client {user_id}] probe_only failed: {e}")
            probe_loss = None

        _fill_common_signals(
            client_signal=client_signal,
            original_seq_len=original_seq_len,
            final_seq_len=original_seq_len,
            last_loss=probe_loss,
            config=config,
            user_seq=user_seq
        )
        
        # Calculate Alignment Signal
        if global_aux_data and "global_update_vec" in global_aux_data:
            probe_update = {k: probe_model.state_dict()[k] - global_state_dict[k] for k in global_state_dict}
            align_score = compute_cosine_similarity(probe_update, global_aux_data["global_update_vec"])
            client_signal["update_quality"]["alignment"] = align_score
        else:
            client_signal["update_quality"]["alignment"] = 1.0

        return clone_state_dict(global_state_dict), 0, client_signal

    # --------------------------------------------------
    # 3. Initialize base model for Training / Generation
    # --------------------------------------------------
    base_model = SASRec(
        num_items=num_items,
        max_seq_len=config["max_seq_len"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    base_model.load_state_dict(global_state_dict, strict=True)

    # --------------------------------------------------
    # 4. ADAPT: Generator + Verifier (Budget Aware)
    # --------------------------------------------------
    generated_items = []
    
    # Chỉ chạy nếu server cấp budget > 0 và config cho phép
    if config.get("use_openai_agents", False) and generation_budget > 0:
        
        # --- A. State-Aware Prompt Routing ---
        diversity_score = compute_diversity_ratio(user_seq)
        seq_len = len(user_seq)
        
        if seq_len < config.get("short_seq_threshold", 20):
            regime = "cold_start"
        elif diversity_score < config.get("diversity_threshold", 0.5):
            regime = "exploration"
        else:
            regime = "standard"
            
        client_signal["adapt_meta"]["regime"] = regime

        # --- B. Generator (OpenAI) ---
        topk_items = get_model_topk(
            base_model,
            user_seq,
            num_items=num_items,
            k=config.get("agent_model_topk", 10),
            device=device,
        )

        user_profile = {
            "recent_items": user_seq[-config.get("agent_recent_k", 10):],
            "seq_len": seq_len,
            "model_topk": topk_items,
        }

        try:
            gen_model = config.get("generator_model", config.get("openai_model", "gpt-3.5-turbo"))
            gen_agent = OpenAIGeneratorAgent(model=gen_model)
            
            raw_generated = gen_agent.run(
                user_profile,
                regime=regime, 
                num_items=num_items,
                hard_cap=generation_budget 
            )
            
            # --- C. Verifier (Training-based) ---
            if len(raw_generated) > 0:
                # Init verifier
                verifier = TrainingVerifier(
                    config=config,
                    device=device,
                    debug=config.get("agent_debug", False)
                )
                
                # [SỬA ĐỔI] Gọi hàm verify với đúng 4 tham số yêu cầu
                accepted_items = verifier.verify(
                    base_model,      # Model để clone và train thử
                    user_seq,        # Lịch sử gốc
                    raw_generated,   # Item cần kiểm tra
                    num_items        # Tổng số item
                )
                
                if len(accepted_items) > 0:
                    user_seq = user_seq + accepted_items
                    client_signal["adapt_meta"]["accepted"] = True
                    client_signal["adapt_meta"]["n_synth"] = len(accepted_items)
                    
                    if config.get("agent_debug", False):
                        print(f"[Client {user_id} | {regime}] Gen {len(raw_generated)} -> Accept {accepted_items}")
                else:
                    client_signal["adapt_meta"]["accepted"] = False
                    if config.get("agent_debug", False):
                        print(f"[Client {user_id} | {regime}] Gen {len(raw_generated)} -> Reject ALL")

        except Exception as e:
            if config.get("agent_debug", False):
                print(f"[Client {user_id}] Agent Error: {e}")

    # --------------------------------------------------
    # 5. Build FINAL dataset & dataloader
    # --------------------------------------------------
    dataset = UserSequenceDataset(
        user_seq=user_seq,
        max_seq_len=config["max_seq_len"],
        num_items=num_items,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("local_batch_size", 32),
        shuffle=True,
        drop_last=False,
    )
    num_samples = len(dataset)

    # --------------------------------------------------
    # 6. Full local training
    # --------------------------------------------------
    model = base_model 
    model.train()

    optimizer = Adam(
        model.parameters(),
        lr=config["local_lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    last_loss: Optional[float] = None
    for _epoch in range(config["local_epochs"]):
        for batch in dataloader:
            input_seq = batch["input_seq"].to(device)
            pos_item = batch["pos_item"].to(device)
            neg_item = batch["neg_item"].to(device)

            pos_items = pos_item.expand(-1, input_seq.size(1))
            neg_items = neg_item.expand(-1, input_seq.size(1))

            optimizer.zero_grad()
            pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
            loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    # --------------------------------------------------
    # 7. Build signals (post-training)
    # --------------------------------------------------
    new_state_dict = clone_state_dict(model.state_dict())

    _fill_common_signals(
        client_signal=client_signal,
        original_seq_len=original_seq_len,
        final_seq_len=len(user_seq),
        last_loss=last_loss,
        config=config,
        user_seq=user_seq
    )

    if global_aux_data and "global_update_vec" in global_aux_data:
         local_update = {k: new_state_dict[k] - global_state_dict[k] for k in global_state_dict}
         align_score = compute_cosine_similarity(local_update, global_aux_data["global_update_vec"])
         client_signal["update_quality"]["alignment"] = align_score
    
    client_signal["update_quality"]["update_norm"] = compute_update_norm(new_state_dict, global_state_dict)

    return new_state_dict, num_samples, client_signal