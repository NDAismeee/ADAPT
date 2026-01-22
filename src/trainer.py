import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from datasets import SASRecTrainDataset, SASRecEvalDataset
from model_sasrec import SASRec, sasrec_bce_loss
from evaluation import evaluate


class Trainer:
    def __init__(
        self,
        model: SASRec,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 50,
        eval_k: int = 20,
        log_interval: int = 100,
        early_stop_patience: int = 5,
        ckpt_dir: str = "outputs/runs",
        lr_step: int = 20,
        lr_gamma: float = 0.5,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_gamma)

        self.epochs = epochs
        self.eval_k = eval_k
        self.log_interval = log_interval
        self.early_stop_patience = early_stop_patience

        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

        self.best_ndcg = -1.0
        self.no_improve_epochs = 0

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        start = time.time()

        for step, batch in enumerate(self.train_loader, start=1):
            input_seq = batch["input_seq"].to(self.device)
            pos_items = batch["pos_items"].to(self.device)
            neg_items = batch["neg_items"].to(self.device)

            self.optimizer.zero_grad()
            pos_logits, neg_logits = self.model(input_seq, pos_items, neg_items)
            loss = sasrec_bce_loss(pos_logits, neg_logits, pos_items)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % self.log_interval == 0:
                print(
                    f"[Epoch {epoch}] Step {step}/{len(self.train_loader)} "
                    f"Loss {loss.item():.4f}"
                )

        self.scheduler.step()
        avg_loss = total_loss / max(1, len(self.train_loader))
        elapsed = time.time() - start
        print(f"[Epoch {epoch}] Train Loss {avg_loss:.4f} | Time {elapsed:.1f}s")
        return avg_loss

    @torch.no_grad()
    def _evaluate(self, split: str = "valid") -> Dict[str, float]:
        loader = self.valid_loader if split == "valid" else self.test_loader
        metrics = evaluate(
            model=self.model,
            dataloader=loader,
            device=self.device,
            k=self.eval_k,
        )
        return metrics

    def _save_ckpt(self):
        torch.save(self.model.state_dict(), self.ckpt_path)

    def _load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))

    def fit(self):
        print("===== Training SASRec =====")
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)

            valid_metrics = self._evaluate(split="valid")
            hr = valid_metrics[f"HR@{self.eval_k}"]
            ndcg = valid_metrics[f"NDCG@{self.eval_k}"]

            print(
                f"[Epoch {epoch}] Valid HR@{self.eval_k} {hr:.4f} "
                f"NDCG@{self.eval_k} {ndcg:.4f}"
            )

            # Early stopping on NDCG
            if ndcg > self.best_ndcg:
                self.best_ndcg = ndcg
                self.no_improve_epochs = 0
                self._save_ckpt()
                print(f"  ✔ New best model saved (NDCG@{self.eval_k}={ndcg:.4f})")
            else:
                self.no_improve_epochs += 1
                print(
                    f"  ✖ No improvement ({self.no_improve_epochs}/"
                    f"{self.early_stop_patience})"
                )

            if self.no_improve_epochs >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

        print("===== Training Finished =====")
        print("Loading best checkpoint and evaluating on test set...")
        self._load_ckpt()
        test_metrics = self._evaluate(split="test")
        print(
            f"Test HR@{self.eval_k} {test_metrics[f'HR@{self.eval_k}']:.4f} | "
            f"NDCG@{self.eval_k} {test_metrics[f'NDCG@{self.eval_k}']:.4f}"
        )
        return test_metrics


# ---------- Helper to quickly build loaders ----------
def build_loaders(
    data_dir: str,
    max_seq_len: int,
    num_items: int,
    train_bs: int = 128,
    eval_bs: int = 256,
    num_workers: int = 2,
):
    train_ds = SASRecTrainDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        num_items=num_items,
    )
    valid_ds = SASRecEvalDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        num_items=num_items,
        mode="valid",
        num_negatives=100,
    )
    test_ds = SASRecEvalDataset(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        num_items=num_items,
        mode="test",
        num_negatives=100,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader
