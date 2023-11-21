"""
Test case for assessing the performance of the model's prediction function.

This test case evaluates the inference time of the model to ensure it meets
performance criteria.
"""
import time
import unittest

from src.predict import predict_model
from tests.test_utils import create_mock_dataloader, create_model


class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        # Assuming num_classes is known, and a mock DataLoader is available
        self.num_classes = 10  # Example number of classes
        self.mock_dataloader = create_mock_dataloader()
        self.model = create_model()

    def test_inference_time(self):
        """
        Test the inference time of the predict_model function.

        This test checks if the time taken for inference is below a specified threshold,
        ensuring that the model's prediction speed is within acceptable limits.
        """
        threshold = 20000  # Large threshold as we are testing on CPU on GitHub Actions
        start_time = time.time()

        labels = {"PF00001": 0, "PF00002": 1, "unk": 2, "pad": 3}
        predict_model(self.model, self.mock_dataloader, labels)

        end_time = time.time()
        self.assertTrue((end_time - start_time) < threshold)


if __name__ == "__main__":
    unittest.main()
