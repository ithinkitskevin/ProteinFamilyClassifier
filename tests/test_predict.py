import unittest
import torch
from src.predict import predict_model
from src.evaluate import evaluate_model
from src.model import ProtCNN
from src.dataset import SequenceDataset
from torch.utils.data import DataLoader
import pandas as pd
from tests.test_utils import create_mock_dataloader, create_model 

class TestPredict(unittest.TestCase):

    def setUp(self):
        self.num_classes = 10  # Example number of classes
        self.mock_dataloader = create_mock_dataloader()
        self.model = create_model()
    
    def test_predict_model(self):
        # Example test for the predict_model function
        # We will use the model and a single batch from the mock_dataloader

        # Perform the prediction
        labels = {"PF00001":0, "PF00002":1, "unk":2, "pad":3}
        predictions = predict_model(self.model, self.mock_dataloader, labels)
        
        # Check that predictions have the correct shape
        self.assertEqual(len(predictions), 4)

        # Optionally, check the range of predictions
        # Here you should define what range is acceptable for your predictions
        self.assertTrue(all(pred in labels for pred in predictions))
        
if __name__ == '__main__':
    unittest.main()
