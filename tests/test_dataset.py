import unittest
from src.dataset import ProteinSequenceDataset

class TestProteinSequenceDataset(unittest.TestCase):

    def test_dataset_sample(self):
        # Test retrieval of a sample from the dataset
        dataset = ProteinSequenceDataset(sequences=["sample1", "sample2"], labels=[0, 1])
        self.assertEqual(len(dataset), 2)

if __name__ == '__main__':
    unittest.main()