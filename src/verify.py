# verify.py
# =========================================================
# ADAPT Verifier (Training-based / Loss-based)
#
# Paper alignment:
#   - Evaluates synthetic data utility by "Look-ahead" training.
#   - Logic: 
#       1. Split user history into Local Train & Local Valid.
#       2. Compute Base Validation Loss.
#       3. For each candidate item:
#           a. Clone the model.
#           b. Train clone on (History + Candidate).
#           c. Compute New Validation Loss.
#           d. Accept if New Loss <= Base Loss.
#
#   - Removes dependency on Local LLM (which cannot verify IDs).
# =========================================================

from __future__ import annotations

import copy
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from fl_datasets import UserSequenceDataset
from model_sasrec import sasrec_bce_loss

class TrainingVerifier:
    """
    Filters generated items based on their empirical impact on 
    local validation loss.
    """

    def __init__(
        self, 
        config: Dict[str, Any], 
        device: torch.device,
        debug: bool = False
    ):
        self.config = config
        self.device = device
        self.debug = debug
        
        # Hyperparameters for verification training
        # Sử dụng LR và batch size riêng cho việc verify để đảm bảo tốc độ
        self.verify_lr = config.get("verify_lr", 1e-3)
        self.verify_batch_size = config.get("verify_batch_size", 16)
        self.max_seq_len = config["max_seq_len"]

    def _get_val_loss(self, model, val_seq: List[int], num_items: int) -> float:
        """
        Compute validation loss on the held-out portion of user history.
        """
        model.eval()
        if len(val_seq) < 2:
            return 0.0
            
        ds = UserSequenceDataset(
            user_seq=val_seq,
            max_seq_len=self.max_seq_len,
            num_items=num_items
        )
        dl = DataLoader(ds, batch_size=self.verify_batch_size, shuffle=False)
        
        total_loss = 0.0
        steps = 0
        
        with torch.no_grad():
            for batch in dl:
                input_seq = batch["input_seq"].to(self.device)
                pos_item = batch["pos_item"].to(self.device)
                neg_item = batch["neg_item"].to(self.device)
                
                pos_items = pos_item.expand(-1, input_seq.size(1))
                neg_items = neg_item.expand(-1, input_seq.size(1))

                pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
                loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
                
                total_loss += loss.item()
                steps += 1
        
        return total_loss / max(1, steps)

    def _train_one_step(self, model, train_seq: List[int], num_items: int):
        """
        Perform one lightweight training step on the augmented sequence.
        """
        model.train()
        optimizer = Adam(model.parameters(), lr=self.verify_lr)
        
        ds = UserSequenceDataset(
            user_seq=train_seq,
            max_seq_len=self.max_seq_len,
            num_items=num_items
        )
        dl = DataLoader(ds, batch_size=self.verify_batch_size, shuffle=True)
        
        # Train for just 1 step to check gradient direction
        for batch in dl:
            input_seq = batch["input_seq"].to(self.device)
            pos_item = batch["pos_item"].to(self.device)
            neg_item = batch["neg_item"].to(self.device)
            
            pos_items = pos_item.expand(-1, input_seq.size(1))
            neg_items = neg_item.expand(-1, input_seq.size(1))

            optimizer.zero_grad()
            pos_logits, neg_logits = model(input_seq, pos_items, neg_items)
            loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
            loss.backward()
            optimizer.step()
            
            break # Stop after 1 batch for speed

    def verify(
        self, 
        model, 
        user_seq: List[int], 
        candidate_items: List[int],
        num_items: int
    ) -> List[int]:
        """
        Main verification loop.
        Input:
            model: Current local model
            user_seq: Real user history
            candidate_items: List of generated IDs
            num_items: Total item count
        Output:
            List of accepted item IDs
        """
        if not candidate_items:
            return []
            
        # 1. Create Local Split (Train / Valid)
        # Nếu lịch sử quá ngắn (<5 item), không thể split an toàn
        # Ta chấp nhận heuristic: Nếu ngắn quá thì tin tưởng Generator (chấp nhận hết)
        n = len(user_seq)
        if n < 5: 
            return candidate_items 

        # Split 80% train, 20% valid
        split_idx = int(n * 0.8)
        real_train = user_seq[:split_idx]
        real_valid = user_seq[split_idx:] 

        # 2. Compute Base Loss (Loss trước khi thêm data giả)
        base_loss = self._get_val_loss(model, real_valid, num_items)
        
        if self.debug:
            print(f"   [Verifier] Base Loss: {base_loss:.4f} | Checking {len(candidate_items)} items")

        accepted_items = []

        # 3. Verify each candidate
        for item in candidate_items:
            # Clone model để không ảnh hưởng model gốc
            temp_model = copy.deepcopy(model)
            
            # Augment: Thêm item vào cuối tập train giả định
            aug_train = real_train + [item]
            
            try:
                # Train thử 1 bước
                self._train_one_step(temp_model, aug_train, num_items)
                
                # Tính Loss mới trên cùng tập Validation gốc
                new_loss = self._get_val_loss(temp_model, real_valid, num_items)
                
                # Acceptance Rule: Loss giảm hoặc tăng không đáng kể (<= 5%)
                if new_loss <= base_loss * 1.05:
                    accepted_items.append(item)
                    if self.debug:
                        print(f"     -> Item {item}: Loss {new_loss:.4f} (Accepted)")
                else:
                    if self.debug:
                        print(f"     -> Item {item}: Loss {new_loss:.4f} (Rejected - Loss increased)")
                        
            except Exception as e:
                if self.debug:
                    print(f"     -> Item {item}: Error {e}")
                continue

        return accepted_items