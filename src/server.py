import os
import csv
import time
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import DataLoader

from model_sasrec import SASRec
from evaluation import evaluate
from datasets import SASRecEvalDataset
from fl_utils import (
    fedavg,
    sample_clients,
    clone_state_dict,
    compute_state_dict_diff  # [MỚI] Hàm tính hiệu 2 state_dict
)
from client import client_update
from trigger_agent import TriggerAgent


class FederatedServer:
    """
    Federated Learning Server for SASRec (ADAPT Framework)
    
    Key Features:
      - Closed-loop Agentic Control: Probe -> Decide Budget -> Train.
      - Global Signal Tracking: Maintains global update vector for alignment calculation.
      - Budget-Aware: Allocates specific generation counts (k_i) to clients.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[int]],
        valid_items: Dict[int, int],
        test_items: Dict[int, int],
        num_items: int,
        config: Dict,
        device: torch.device,
        ckpt_dir: str = "outputs/fl_runs",
    ):
        self.train_seqs = train_seqs
        self.valid_items = valid_items
        self.test_items = test_items
        self.num_items = num_items
        self.config = config
        self.device = device

        self.users = list(train_seqs.keys())

        # --------------------------------------------------
        # Global model
        # --------------------------------------------------
        self.global_model = SASRec(
            num_items=num_items,
            max_seq_len=config["max_seq_len"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(device)

        self.global_state = clone_state_dict(self.global_model.state_dict())
        
        # [ADAPT] Track previous global state to compute global update direction
        # Initialize with zeros (first round has no previous direction)
        self.last_global_update = None 

        # --------------------------------------------------
        # Trigger Agent (Local LLM)
        # --------------------------------------------------
        self.trigger_agent = TriggerAgent(
            model_name=config.get("local_llm_path", "Qwen/Qwen2.5-0.5B-Instruct"),
            debug=config.get("agent_debug", False)
        )

        # --------------------------------------------------
        # Evaluation datasets & loaders
        # --------------------------------------------------
        self.valid_dataset = SASRecEvalDataset(
            data_dir=config["data_dir"],
            max_seq_len=config["max_seq_len"],
            num_items=num_items,
            mode="valid",
            num_negatives=config.get("num_negatives", 100),
        )
        self.test_dataset = SASRecEvalDataset(
            data_dir=config["data_dir"],
            max_seq_len=config["max_seq_len"],
            num_items=num_items,
            mode="test",
            num_negatives=config.get("num_negatives", 100),
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
        )

        # --------------------------------------------------
        # Checkpoint & tracking
        # --------------------------------------------------
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(ckpt_dir, "best_global_model.pt")
        self.best_ndcg = -1.0

        # --------------------------------------------------
        # CSV logging
        # --------------------------------------------------
        self.csv_path = os.path.join(ckpt_dir, "adapt_round_metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "num_selected_clients",
                "num_triggered_clients",
                "avg_budget", # [MỚI] Track average budget allocated
                f"HR@{self.config.get('eval_k', 20)}",
                f"NDCG@{self.config.get('eval_k', 20)}",
                "round_time_seconds"
            ])

    # ==================================================
    # One Federated Round
    # ==================================================
    def run_round(self, round_idx: int) -> Tuple[int, int, float, float]:
        print(f"\n========== FL Round {round_idx} ==========")
        start_time = time.time()

        # ----------------------------------------------
        # 1. Sample clients
        # ----------------------------------------------
        selected_users = sample_clients(
            all_users=self.users,
            clients_per_round=self.config.get("clients_per_round"),
            client_fraction=self.config.get("client_fraction"),
        )

        num_selected = len(selected_users)
        print(f"Selected {num_selected} clients")

        client_states = []
        client_weights = []
        
        # Track triggering stats
        num_triggered = 0
        total_budget_allocated = 0

        # [ADAPT] Prepare auxiliary data for clients (Global Update Vector)
        global_aux_data = {}
        if self.last_global_update is not None:
            global_aux_data["global_update_vec"] = self.last_global_update

        # ----------------------------------------------
        # 2. Probe -> Decide -> Train
        # ----------------------------------------------
        for user_id in selected_users:
            user_seq = self.train_seqs[user_id]

            # ---- STEP 1: PROBE (Get Signals) ----
            # Client calculates loss, diversity, and alignment (using global_aux_data)
            _, _, probe_signal = client_update(
                user_id=user_id,
                user_seq=user_seq,
                global_state_dict=self.global_state,
                num_items=self.num_items,
                config=self.config,
                device=self.device,
                probe_only=True,
                global_aux_data=global_aux_data
            )

            # ---- STEP 2: DECIDE (Agent allocates Budget) ----
            # Agent returns an integer budget (e.g., 0, 3, 5 items)
            budget = self.trigger_agent.run(probe_signal)
            
            if budget > 0:
                num_triggered += 1
                total_budget_allocated += budget

            # ---- STEP 3: TRAIN (Execute with Budget) ----
            new_state, num_samples, _ = client_update(
                user_id=user_id,
                user_seq=user_seq,
                global_state_dict=self.global_state,
                num_items=self.num_items,
                config=self.config,
                device=self.device,
                generation_budget=budget, # [ADAPT] Pass budget instead of bool
                probe_only=False,
                global_aux_data=global_aux_data
            )

            if num_samples > 0:
                client_states.append(new_state)
                client_weights.append(num_samples)

        if len(client_states) == 0:
            print("⚠️ No valid client updates in this round")
            return 0, num_selected, 0.0, 0.0

        # ----------------------------------------------
        # 3. FedAvg aggregation
        # ----------------------------------------------
        # Backup current global state before updating
        old_global_state = clone_state_dict(self.global_state)
        
        # Aggregate
        self.global_state = fedavg(client_states, client_weights)
        self.global_model.load_state_dict(self.global_state, strict=True)

        # [ADAPT] Compute Global Update Vector for NEXT round
        # \Delta \theta_{global} = \theta_{new} - \theta_{old}
        self.last_global_update = compute_state_dict_diff(self.global_state, old_global_state)

        # Timing
        end_time = time.time()
        round_duration = end_time - start_time
        avg_budget = total_budget_allocated / max(1, num_triggered) if num_triggered > 0 else 0.0
        
        print(f"⏱️ Round {round_idx} finished in {round_duration:.2f}s")
        print(f"   Triggered: {num_triggered}/{num_selected}, Avg Budget: {avg_budget:.1f}")

        return num_triggered, num_selected, round_duration, avg_budget

    # ==================================================
    # Evaluation
    # ==================================================
    @torch.no_grad()
    def evaluate_global(self, split: str = "valid"):
        self.global_model.eval()
        loader = self.valid_loader if split == "valid" else self.test_loader
        return evaluate(
            model=self.global_model,
            dataloader=loader,
            device=self.device,
            k=self.config.get("eval_k", 20),
        )

    # ==================================================
    # Training Loop
    # ==================================================
    def fit(self):
        num_rounds = self.config["num_rounds"]
        eval_every = self.config.get("eval_every", 1)

        print("===== Start Federated Training (ADAPT: Budget-Aware Agent) =====")

        for r in range(1, num_rounds + 1):
            num_triggered, num_selected, duration, avg_budget = self.run_round(r)

            if r % eval_every == 0:
                valid_metrics = self.evaluate_global(split="valid")
                hr = valid_metrics[f"HR@{self.config.get('eval_k', 20)}"]
                ndcg = valid_metrics[f"NDCG@{self.config.get('eval_k', 20)}"]

                trigger_ratio = num_triggered / max(1, num_selected)

                print(
                    f"[Round {r}] "
                    f"HR={hr:.4f} | NDCG={ndcg:.4f} | "
                    f"Trig={num_triggered} ({trigger_ratio:.0%}) | "
                    f"Budg={avg_budget:.1f}"
                )

                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        r,
                        num_selected,
                        num_triggered,
                        round(avg_budget, 2),
                        round(hr, 6),
                        round(ndcg, 6),
                        round(duration, 4)
                    ])

                if ndcg > self.best_ndcg:
                    self.best_ndcg = ndcg
                    torch.save(self.global_state, self.ckpt_path)
                    print(f"  ✔ New best global model saved (NDCG={ndcg:.4f})")

        # --------------------------------------------------
        # Final test evaluation
        # --------------------------------------------------
        print("\n===== Federated Training Finished =====")
        print("Evaluating best global model on TEST set...")

        self.global_state = torch.load(self.ckpt_path, map_location=self.device)
        self.global_model.load_state_dict(self.global_state, strict=True)

        test_metrics = self.evaluate_global(split="test")
        print(
            f"Test HR@{self.config.get('eval_k', 20)} "
            f"{test_metrics[f'HR@{self.config.get('eval_k', 20)}']:.4f} | "
            f"NDCG@{self.config.get('eval_k', 20)} "
            f"{test_metrics[f'NDCG@{self.config.get('eval_k', 20)}']:.4f}"
        )

        return test_metrics