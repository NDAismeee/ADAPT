import os
import argparse
import pickle
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Preprocess MovieLens-1M for SASRec")
    parser.add_argument("--data_dir", type=str, default="../dataset/ml-1m",
                        help="Path to raw ML-1M data")
    parser.add_argument("--output_dir", type=str, default="../dataset/ml-1m/processed",
                        help="Path to save processed data")
    parser.add_argument("--min_interactions", type=int, default=20,
                        help="Minimum number of interactions per user")
    return parser.parse_args()


def load_ratings(data_dir):
    """
    ratings.dat format:
    UserID::MovieID::Rating::Timestamp
    """
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df


def filter_users(df, min_interactions):
    """
    Keep users with enough interactions
    """
    user_cnt = df.groupby("user_id").size()
    valid_users = user_cnt[user_cnt >= min_interactions].index
    return df[df["user_id"].isin(valid_users)]


def remap_ids(df):
    """
    Remap ids to consecutive integers
    - user_id: 0 ... num_users-1
    - item_id: 1 ... num_items (0 reserved for padding)
    """
    user2id = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2id = {i: j + 1 for j, i in enumerate(df["item_id"].unique())}

    df["user_id"] = df["user_id"].map(user2id)
    df["item_id"] = df["item_id"].map(item2id)

    return df, user2id, item2id


def build_sequences(df):
    """
    Build user interaction sequences sorted by timestamp
    """
    user_seqs = defaultdict(list)

    df = df.sort_values(["user_id", "timestamp"])
    for row in df.itertuples(index=False):
        user_seqs[row.user_id].append(row.item_id)

    return user_seqs


def split_leave_one_out(user_seqs):
    """
    Leave-one-out split (time-aware)
    - train: all except last 2
    - valid: second last
    - test : last
    """
    train_seqs = {}
    valid_items = {}
    test_items = {}

    for user, seq in user_seqs.items():
        if len(seq) < 3:
            continue

        train_seqs[user] = seq[:-2]
        valid_items[user] = seq[-2]
        test_items[user] = seq[-1]

    return train_seqs, valid_items, test_items


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ratings...")
    df = load_ratings(args.data_dir)

    print("Filtering users...")
    df = filter_users(df, args.min_interactions)

    print("Remapping ids...")
    df, user2id, item2id = remap_ids(df)

    print("Building sequential user histories...")
    user_seqs = build_sequences(df)

    print("Splitting train / valid / test...")
    train_seqs, valid_items, test_items = split_leave_one_out(user_seqs)

    stats = {
        "num_users": len(train_seqs),
        "num_items": len(item2id),
        "num_interactions": sum(len(v) for v in user_seqs.values())
    }

    print("Saving processed data...")
    save_pickle(train_seqs, os.path.join(args.output_dir, "train_seqs.pkl"))
    save_pickle(valid_items, os.path.join(args.output_dir, "valid_items.pkl"))
    save_pickle(test_items, os.path.join(args.output_dir, "test_items.pkl"))
    save_pickle(user2id, os.path.join(args.output_dir, "user2id.pkl"))
    save_pickle(item2id, os.path.join(args.output_dir, "item2id.pkl"))
    save_pickle(stats, os.path.join(args.output_dir, "stats.pkl"))

    print("Done.")
    print(stats)


if __name__ == "__main__":
    main()
