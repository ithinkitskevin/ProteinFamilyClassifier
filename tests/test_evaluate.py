"""
Test case for the evaluate_model function.

This test case verifies if the evaluate_model function correctly computes
and returns performance metrics such as accuracy and F1 score.
"""
import unittest

import pandas as pd

from src.evaluate import evaluate_model
from tests.test_utils import create_mock_dataloader, create_model


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        self.num_classes = 17320
        label_data = pd.Series(["PF00001", "PF00002", "PF00003", "PF00004"])
        self.mock_dataloader = create_mock_dataloader(label_data)
        self.model = create_model()

    def test_evaluate_model(self):
        """
        Test the evaluate_model function to ensure it correctly calculates evaluation metrics.

        This test checks if the returned results from evaluate_model include accuracy and F1 score,
        and verifies that these metrics are within the expected range.
        """

        # Perform the evaluation
        results = evaluate_model(self.model, self.mock_dataloader)
        result = results[0]

        # Check that results is a dictionary containing the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("test_acc", result)
        self.assertIn("test_f1", result)

        # Check that the metrics are within an acceptable range
        self.assertGreaterEqual(result["test_acc"], 0.0)
        self.assertLessEqual(result["test_acc"], 1.0)
        self.assertGreaterEqual(result["test_f1"], 0.0)
        self.assertLessEqual(result["test_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
