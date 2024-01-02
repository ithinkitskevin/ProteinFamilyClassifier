"""
Test case for the predict_model function.

This test case verifies if the predict_model function returns predictions
that are of the correct shape and within the expected range, based on the
mock DataLoader and model.
"""
import unittest

from src.predict import predict_model
from tests.test_utils import create_mock_dataloader, create_model


class TestPredict(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        self.num_classes = 17320  # Example number of classes
        self.mock_dataloader = create_mock_dataloader()
        self.model = create_model()

    def test_predict_model(self):
        """
        Test the predict_model function for generating predictions.

        This test checks if the predictions returned by the predict_model function
        are of the expected length and whether each prediction falls within the
        defined range of labels.
        """
        # Perform the prediction
        labels = {"PF00001": 0, "PF00002": 1, "unk": 2, "pad": 3}
        predictions = predict_model(self.model, self.mock_dataloader, labels)

        # Check that predictions have the correct shape
        self.assertEqual(len(predictions), 4)

        # Optionally, check the range of predictions
        # Here you should define what range is acceptable for your predictions
        self.assertTrue(all(pred in labels for pred in predictions))


if __name__ == "__main__":
    unittest.main()
