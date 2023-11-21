
"""
Creates a mock DataLoader for testing purposes.
"""
    
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import SequenceDataset
from src.model import ProtCNN


def create_mock_dataloader(label=None):
    """
    This function generates a DataLoader with predefined sequence data,
    which can be used for testing models and functions that require DataLoader input.

    Args:
        label (Optional[Iterable]): Labels corresponding to each sequence. Default is None.

    Returns:
        DataLoader: A DataLoader instance with mock data.
    """
    # Create DataFrame for sequences
    sequences = pd.Series(
        [
            "AFLFSGRREVMADACLQGMMGCVYGTAGGMDSAAAVLGDFCFLAGK",
            "MVDVGGKPVSRRTAAASATVLLGEKAFWLVKENQLAKGDALAVAQI",
            "VLDVACGTCDVAMEARNQTGDAAFIIGTDFSPGMLTLGLQKLKKNR",
            "VVLERASLESVKVGKEYQLLNCDRHKGIAKKFKRDISTCRPDITHQ",
        ]
    )
    # Create DataFrame for labels
    word2id = {
        "A": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "K": 10,
        "L": 11,
        "M": 12,
        "N": 13,
        "P": 14,
        "Q": 15,
        "R": 16,
        "S": 17,
        "T": 18,
        "V": 19,
        "W": 20,
        "Y": 21,
        "<pad>": 0,
        "<unk>": 1,
    }
    # Assuming SequenceDataset expects 'data' and 'label' as DataFrame
    dataset = SequenceDataset(
        word2id=word2id,
        fam2label={"PF00001": 0, "PF00002": 1, "<unk>": 2, "<pad>": 3},
        max_len=120,
        data=sequences,
        label=label,
    )

    test_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    return test_dataloader


def create_model():
    """
    Creates an instance of the ProtCNN model.

    This function is used for testing purposes to generate a model instance
    with predefined configurations.

    Returns:
        ProtCNN: An instance of the ProtCNN model.
    """
    return ProtCNN(num_classes=17320)
