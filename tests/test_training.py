import unittest
from src.train import train_model
from src.model import ProtCNN
from src.dataset import ProteinSequenceDataset

class TestTraining(unittest.TestCase):

    def test_training_pipeline(self):
        # Test the complete training process
        model = ProtCNN(input_dim=100, num_classes=10)
        dataset = ProteinSequenceDataset(sequences=["sample1", "sample2"], labels=[0, 1])
        # Further setup and call train_model
        pass

if __name__ == '__main__':
    unittest.main()
    
    
