import unittest
from src.dataset import SequenceDataset
import pandas as pd

class TestSequenceDataset(unittest.TestCase):

    def test_dataset_sample(self):
        # Test retrieval of a sample from the dataset
        
        # Create DataFrame for sequences
        sequences = pd.Series([
            "AFLFSGRREVMADACLQGMMGCVYGTAGGMDSAAAVLGDFCFLAGK",
            "MVDVGGKPVSRRTAAASATVLLGEKAFWLVKENQLAKGDALAVAQI",
            "VLDVACGTCDVAMEARNQTGDAAFIIGTDFSPGMLTLGLQKLKKNR",
            "VVLERASLESVKVGKEYQLLNCDRHKGIAKKFKRDISTCRPDITHQ"
        ])
        # Create DataFrame for labels
        label_data = pd.Series(["PF00001", "PF00001", "PF00001", "PF00001"])
        word2id = {'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'V': 19, 'W': 20, 'Y': 21, '<pad>': 0, '<unk>': 1}
        # Assuming SequenceDataset expects 'data' and 'label' as DataFrame
        dataset = SequenceDataset(
            word2id=word2id,
            fam2label={"PF00001":0, "PF00002":1, "<unk>":2, "<pad>":3},
            max_len=120,  
            data=sequences,
            label=label_data
        )
        
        self.assertEqual(len(dataset), 4)

if __name__ == '__main__':
    unittest.main()