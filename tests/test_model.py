"""
Test case for the initialization of different models.

This test case ensures that the models can be initialized correctly and
are instances of their respective classes.
"""
import unittest

from src.model import ProtCNN


class TestModel(unittest.TestCase):
    def test_cnn_model_initialization(self):
        """
        Test the initialization of the ProtCNN model.
        """
        model = ProtCNN(num_classes=10)
        self.assertIsInstance(model, ProtCNN)

if __name__ == "__main__":
    unittest.main()
