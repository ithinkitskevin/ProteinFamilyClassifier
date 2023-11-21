"""
Module for handling dataset operations for protein sequence classification.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data

def build_vocab(data):
    """
    Builds a vocabulary from given protein sequences.

    Args:
        data (list of str): List of protein sequences.

    Returns:
        dict: A dictionary mapping each amino acid to a unique integer.
    """
    # Define rare amino acids to be excluded
    rare_aas = {"X", "U", "B", "O", "Z"}

    # Create a set of all unique amino acids excluding the rare ones
    unique_aas = sorted(set().union(*data) - rare_aas)

    # Map each amino acid to a unique integer, starting from 2 (unk/pad tokens)
    word2id = {aa: idx for idx, aa in enumerate(unique_aas, start=2)}

    # Add special tokens for padding and unknown amino acids
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1

    return word2id

def build_labels(targets):
    """
    Builds a label mapping from given target labels.

    Args:
        targets (pandas.Series): Series of target labels.

    Returns:
        dict: A dictionary mapping each unique label to an integer.
    """
    # Map each unique label to a unique integer, starting from 1 (unk token)
    fam2label = {label: idx for idx, label in enumerate(targets.unique(), start=1)}

    # Add a special token for unknown labels
    fam2label["<unk>"] = 0

    return fam2label

def reader(partition, data_path):
    """
    Reads and concatenates data from multiple files in a given directory.

    Args:
        partition (str): The partition of the dataset to read ('train', 'test', 'dev'.).
        data_path (str): Path to the directory containing the dataset.

    Returns:
        tuple: Two pandas.Series, one for sequences and one for family accessions.
    """
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
    """
    A PyTorch Dataset for handling sequence data.

    Attributes:
        word2id (dict): Mapping from amino acids to integers.
        fam2label (dict): Mapping from family labels to integers.
        max_len (int): Maximum length of sequences.
        data (pandas.Series): Series containing sequences.
        label (pandas.Series): Series containing labels.
    """
    def __init__(self, word2id, fam2label, max_len, data, label=None):
        """
        Initializes the SequenceDataset.

        Args:
            word2id (dict): Mapping from amino acids to integers.
            fam2label (dict): Mapping from family labels to integers.
            max_len (int): Maximum length of sequences.
            data (pandas.Series): Series containing sequences.
            label (pandas.Series, optional): Series containing labels. Defaults to None.
        """
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.data = data
        self.label = label

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item by its index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict/torch.Tensor: A dictionary containing the sequence tensor and its label.
                               Or just the sequence tensor.
        """
        seq = self.preprocess(self.data.iloc[index])
        return {'sequence': seq, 'target': self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])} if self.label is not None else seq

    def preprocess(self, text):
        """
        Preprocesses a sequence text into a one-hot tensor.

        Args:
            text (str): The sequence text to preprocess.

        Returns:
            torch.Tensor: A tensor representation of the preprocessed sequence.
        """
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
