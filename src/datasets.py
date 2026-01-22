import random
import pickle
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class SASRecTrainDataset(Dataset):
    """
    Training dataset for SASRec
    Each sample:
      - input_seq : [max_len]
      - pos_items : [max_len]
      - neg_items : [max_len]
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int,
        num_items: int,
    ):
        self.max_seq_len = max_seq_len
        self.num_items = num_items

        self.user_seqs: Dict[int, List[int]] = load_pickle(
            f"{data_dir}/train_seqs.pkl"
        )

        self.users = list(self.user_seqs.keys())

    def __len__(self):
        return len(self.users)

    def _negative_sampling(self, user_seq_set):
        """
        Sample a negative item not interacted by user
        """
        neg = random.randint(1, self.num_items)
        while neg in user_seq_set:
            neg = random.randint(1, self.num_items)
        return neg

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_seqs[user]

        # Input sequence (all but last item)
        input_seq = seq[:-1]
        target_seq = seq[1:]

        seq_len = len(input_seq)

        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            input_seq = [0] * pad_len + input_seq
            target_seq = [0] * pad_len + target_seq
        else:
            input_seq = input_seq[-self.max_seq_len:]
            target_seq = target_seq[-self.max_seq_len:]

        user_seq_set = set(seq)

        neg_seq = []
        for pos in target_seq:
            if pos == 0:
                neg_seq.append(0)
            else:
                neg_seq.append(self._negative_sampling(user_seq_set))

        return {
            "input_seq": torch.LongTensor(input_seq),
            "pos_items": torch.LongTensor(target_seq),
            "neg_items": torch.LongTensor(neg_seq),
        }


class SASRecEvalDataset(Dataset):
    """
    Evaluation dataset (Validation / Test)
    Each sample:
      - input_seq : [max_len]
      - target_item : scalar
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int,
        num_items: int,
        mode: str = "valid",  # or "test"
        num_negatives: int = 100,
    ):
        assert mode in ["valid", "test"]

        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.mode = mode

        self.train_seqs = load_pickle(f"{data_dir}/train_seqs.pkl")
        self.valid_items = load_pickle(f"{data_dir}/valid_items.pkl")
        self.test_items = load_pickle(f"{data_dir}/test_items.pkl")

        self.users = list(self.train_seqs.keys())

    def __len__(self):
        return len(self.users)

    def _sample_negatives(self, user_seq_set, target_item):
        negs = set()
        while len(negs) < self.num_negatives:
            neg = random.randint(1, self.num_items)
            if neg != target_item and neg not in user_seq_set:
                negs.add(neg)
        return list(negs)

    def __getitem__(self, idx):
        user = self.users[idx]
        train_seq = self.train_seqs[user]

        if self.mode == "valid":
            input_seq = train_seq
            target_item = self.valid_items[user]
        else:
            input_seq = train_seq + [self.valid_items[user]]
            target_item = self.test_items[user]

        # pad / truncate using input_seq (NOT train_seq)
        if len(input_seq) >= self.max_seq_len:
            input_seq = input_seq[-self.max_seq_len:]
        else:
            input_seq = [0] * (self.max_seq_len - len(input_seq)) + input_seq

        # negatives should not be in user's known history (input_seq)
        user_seq_set = set(input_seq)
        user_seq_set.discard(0)

        negatives = self._sample_negatives(user_seq_set, target_item)

        candidates = [target_item] + negatives
        labels = [1] + [0] * self.num_negatives

        return {
            "input_seq": torch.LongTensor(input_seq),
            "candidates": torch.LongTensor(candidates),
            "labels": torch.LongTensor(labels),
        }
