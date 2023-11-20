import unittest
from src.train import train_model
from src.model import ProtCNN
from src.dataset import SequenceDataset

class TestTraining(unittest.TestCase):

    def test_training_pipeline(self):
        # Test the complete training process
        model = ProtCNN(num_classes=10)

if __name__ == '__main__':
    unittest.main()
    
    
