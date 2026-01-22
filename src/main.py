import argparse
import pickle
import os
import torch

from model_sasrec import SASRec
from trainer import Trainer, build_loaders


def parse_args():
    parser = argparse.ArgumentParser("Train SASRec on MovieLens-1M")

    # Paths
    parser.add_argument("--data_dir", type=str, default="../dataset/ml-1m/processed",
                        help="Processed data directory")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/runs",
                        help="Checkpoint directory")

    # Model hyperparameters
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=20)

    # Evaluation
    parser.add_argument("--eval_k", type=int, default=20)

    return parser.parse_args()


def load_stats(data_dir):
    stats_path = os.path.join(data_dir, "stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset stats
    stats = load_stats(args.data_dir)
    num_items = stats["num_items"]
    num_users = stats["num_users"]

    print(f"Dataset stats: users={num_users}, items={num_items}")

    # Build dataloaders
    train_loader, valid_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        num_items=num_items,
        train_bs=args.batch_size,
        eval_bs=args.eval_batch_size,
    )

    # Build model
    model = SASRec(
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        eval_k=args.eval_k,
        early_stop_patience=args.early_stop,
        ckpt_dir=args.ckpt_dir,
    )

    # Train + Test
    trainer.fit()


if __name__ == "__main__":
    main()
