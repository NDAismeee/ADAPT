import random
from typing import List

import torch
from torch.utils.data import Dataset


class UserSequenceDataset(Dataset):
    """
    Local dataset for ONE user in Federated Learning.

    From a user sequence:
        [i1, i2, i3, ..., iT]

    Create samples with sliding window:
        input_seq -> predict pos_item

    Each sample:
        - input_seq : (max_seq_len,)
        - pos_item  : scalar
        - neg_item  : scalar
    """

    def __init__(
        self,
        user_seq: List[int],
        max_seq_len: int,
        num_items: int,
    ):
        """
        Args:
            user_seq: list of item ids (train sequence only)
            max_seq_len: max sequence length for SASRec
            num_items: total number of items
        """
        if max(user_seq) >= num_items:
            raise ValueError("Invalid item ID detected in user_seq")

        self.user_seq = user_seq
        self.max_seq_len = max_seq_len
        self.num_items = num_items

        # positions where we can predict next item
        # predict item at position t using history [:t]
        self.indices = list(range(1, len(user_seq)))

        # set for fast negative sampling
        self.user_item_set = set(user_seq)

    def __len__(self):
        return len(self.indices)

    def _negative_sampling(self) -> int:
        """
        Sample a negative item not in user's interaction history
        """
        neg = random.randint(1, self.num_items)
        while neg in self.user_item_set:
            neg = random.randint(1, self.num_items)
        return neg

    def __getitem__(self, idx):
        """
        Returns one training sample
        """
        t = self.indices[idx]

        # history before position t
        hist = self.user_seq[:t]

        # truncate or pad history
        if len(hist) >= self.max_seq_len:
            input_seq = hist[-self.max_seq_len:]
        else:
            input_seq = [0] * (self.max_seq_len - len(hist)) + hist

        pos_item = self.user_seq[t]
        neg_item = self._negative_sampling()

        return {
            "input_seq": torch.LongTensor(input_seq),
            "pos_item": torch.LongTensor([pos_item]),
            "neg_item": torch.LongTensor([neg_item]),
        }
