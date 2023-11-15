import unittest
from src.model import ProtCNN

class TestModel(unittest.TestCase):

    def test_model_initialization(self):
        # Test model initialization
        model = ProtCNN(input_dim=100, num_classes=10)
        self.assertIsInstance(model, ProtCNN)

if __name__ == '__main__':
    unittest.main()