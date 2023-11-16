import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


def build_vocab(data):
    # Build the vocabulary
    voc = set()
    rare_AAs = {"X", "U", "B", "O", "Z"}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1

    return word2id


def build_labels(targets):
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label["<unk>"] = 0
    return fam2label


def reader(partition, data_path):
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(
                pd.read_csv(
                    file, index_col=None, usecols=["sequence", "family_accession"]
                )
            )

    all_data = pd.concat(data)
    return all_data["sequence"], all_data["family_accession"]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, word2id, fam2label, max_len, data, label):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label["<unk>"])

        return {"sequence": seq, "target": label}

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[: self.max_len]:
            seq.append(self.word2id.get(word, self.word2id["<unk>"]))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id["<pad>"] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(
            seq,
            num_classes=len(self.word2id),
        )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq
