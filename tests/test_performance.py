import unittest
import time
from src.model import ProtCNN

class TestModelPerformance(unittest.TestCase):

    def test_inference_time(self):
        model = ProtCNN(input_dim=100, num_classes=10)
        start_time = time.time()
        # Run model inference
        end_time = time.time()
        self.assertTrue((end_time - start_time) < some_acceptable_threshold)

if __name__ == '__main__':
    unittest.main()