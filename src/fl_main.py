import os
import argparse
import pickle
import torch
import numpy as np
import random

from fl_utils import set_seed
from server import FederatedServer
from dotenv import load_dotenv

# Load environment variables (API Keys, etc.)
load_dotenv()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser("Federated SASRec (FedAvg) - ADAPT Framework")

    # --- Data Arguments ---
    parser.add_argument("--data_dir", type=str, default="../dataset/ml-1m/processed",
                        help="Directory containing processed *.pkl files")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/fl_runs",
                        help="Directory to save checkpoints and logs")

    # --- Federated Learning Arguments ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of FL rounds")
    parser.add_argument("--clients_per_round", type=int, default=20, help="Clients sampled per round")
    parser.add_argument("--client_fraction", type=float, default=None, 
                        help="If set, overrides clients_per_round (e.g., 0.1)")

    # --- Local Training Arguments ---
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--local_batch_size", type=int, default=64)
    parser.add_argument("--local_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # --- Model Architecture Arguments (SASRec) ---
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # --- Evaluation Arguments ---
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_k", type=int, default=20)
    parser.add_argument("--num_negatives", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N rounds")

    # --- ADAPT Agent Arguments ---
    # Generator (Client-side)
    parser.add_argument("--use_openai_agents", action="store_true", default=True,
                        help="Enable OpenAI Generator on clients")
    parser.add_argument("--openai_model", type=str, default="gpt-5-nano",
                        help="OpenAI model for generation (e.g. gpt-3.5-turbo, gpt-4)")
    
    # Trigger/Verifier (Server & Client side Local Models)
    parser.add_argument("--local_llm_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace path for the Local LLM used by Trigger Agent")
    
    # Thresholds for Signals
    parser.add_argument("--short_seq_threshold", type=int, default=20,
                        help="Signal: Sequence length below this is considered 'Cold Start'")
    parser.add_argument("--high_loss_threshold", type=float, default=0.6,
                        help="Signal: Loss above this is considered 'High Loss'")
    parser.add_argument("--diversity_threshold", type=float, default=0.5,
                        help="Signal: Diversity ratio below this triggers 'Exploration' regime")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 1. Load Processed Data
    # ----------------------------
    print(f"Loading data from {args.data_dir}...")
    try:
        train_seqs = load_pickle(os.path.join(args.data_dir, "train_seqs.pkl"))
        valid_items = load_pickle(os.path.join(args.data_dir, "valid_items.pkl"))
        test_items = load_pickle(os.path.join(args.data_dir, "test_items.pkl"))
        stats = load_pickle(os.path.join(args.data_dir, "stats.pkl"))
    except FileNotFoundError:
        print(f"Error: Data files not found in {args.data_dir}. Please run preprocess_ml1m.py first.")
        return

    num_items = stats["num_items"]
    num_users = stats["num_users"]
    print(f"Data loaded: {num_users} users, {num_items} items.")

    # ----------------------------
    # 2. Build Configuration Dictionary
    # ----------------------------
    # This config is passed to Server -> Client -> Agents
    config = {
        # System
        "data_dir": args.data_dir,
        "device": str(device),
        
        # Federated Learning
        "num_rounds": args.num_rounds,
        "clients_per_round": args.clients_per_round,
        "client_fraction": args.client_fraction,

        # Local Training
        "local_epochs": args.local_epochs,
        "local_batch_size": args.local_batch_size,
        "local_lr": args.local_lr,
        "weight_decay": args.weight_decay,

        # SASRec Model
        "max_seq_len": args.max_seq_len,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,

        # Evaluation
        "eval_batch_size": args.eval_batch_size,
        "eval_k": args.eval_k,
        "num_negatives": args.num_negatives,
        "eval_every": args.eval_every,

        # --- ADAPT AGENT CONFIG ---
        "use_openai_agents": args.use_openai_agents,
        "openai_model": args.openai_model,
        "local_llm_path": args.local_llm_path,
        
        # Generation Logic
        "agent_recent_k": 10,           # User profile sends last k items
        "agent_model_topk": 10,         # User profile sends top k model predictions
        
        # Signal Thresholds (Important for Client Logic)
        "short_seq_threshold": args.short_seq_threshold,
        "high_loss_threshold": args.high_loss_threshold,
        "diversity_threshold": args.diversity_threshold,
        
        # Verification Logic (New Params for TrainingVerifier)
        "verify_lr": 1e-3,              # Learning rate for the look-ahead verifier
        "verify_batch_size": 16,        # Batch size for verification
        
        # Debugging
        "agent_debug": True,            # Print agent decisions to console
    }

    # ----------------------------
    # 3. Initialize and Run Server
    # ----------------------------
    server = FederatedServer(
        train_seqs=train_seqs,
        valid_items=valid_items,
        test_items=test_items,
        num_items=num_items,
        config=config,
        device=device,
        ckpt_dir=args.ckpt_dir,
    )

    print("Starting ADAPT Federated Training...")
    server.fit()


if __name__ == "__main__":
    main()