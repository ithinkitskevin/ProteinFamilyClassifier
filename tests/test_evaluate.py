import unittest
import torch
from src.evaluate import evaluate_model
from src.model import ProtCNN
from src.dataset import SequenceDataset
from torch.utils.data import DataLoader
import pandas as pd
import string
from tests.test_utils import create_mock_dataloader, create_model 

class TestEvaluate(unittest.TestCase):

    def setUp(self):
        # Assuming num_classes is known, and a mock DataLoader is available
        self.num_classes = 10  # Example number of classes
        
        label_data = pd.Series(["PF00001", "PF00002", "PF00003", "PF00004"])
        self.mock_dataloader = create_mock_dataloader(label_data)
        self.model = create_model()

    def test_evaluate_model(self):
        # Example test for the evaluate_model function

        # Perform the evaluation
        results = evaluate_model(self.model, self.mock_dataloader)
        result = results[0]
        
        # Check that results is a dictionary containing the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('test_acc', result)
        self.assertIn('test_f1', result)

        # Check that the metrics are within an acceptable range
        # Here you should define what is an "acceptable" range for your use case
        self.assertGreaterEqual(result['test_acc'], 0.0)
        self.assertLessEqual(result['test_acc'], 1.0)
        self.assertGreaterEqual(result['test_f1'], 0.0)
        self.assertLessEqual(result['test_f1'], 1.0)

if __name__ == '__main__':
    unittest.main()
