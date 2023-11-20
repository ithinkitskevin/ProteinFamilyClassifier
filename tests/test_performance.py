import unittest
import time
from src.model import ProtCNN
import pandas as pd
import string
from tests.test_utils import create_mock_dataloader, create_model 
from src.predict import predict_model

class TestModelPerformance(unittest.TestCase):

    def setUp(self):
        # Assuming num_classes is known, and a mock DataLoader is available
        self.num_classes = 10  # Example number of classes
        self.mock_dataloader = create_mock_dataloader()
        self.model = create_model()
        
    def test_inference_time(self):
        threshold = 20000  # Large threshold as we are testing on CPU on GitHub Actions
        start_time = time.time()
        
        labels = {"PF00001":0, "PF00002":1, "unk":2, "pad":3}
        predict_model(self.model, self.mock_dataloader, labels)
        
        end_time = time.time()
        self.assertTrue((end_time - start_time) < threshold)

if __name__ == '__main__':
    unittest.main()