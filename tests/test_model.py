import unittest
from src.model import ProtCNN, ProteinTransformer

class TestModel(unittest.TestCase):

    def test_cnn_model_initialization(self):
        # Test model initialization
        model = ProtCNN(num_classes=10)
        self.assertIsInstance(model, ProtCNN)
        
    def test_transformer_model_initialization(self):
        # Test model initialization
        model = ProteinTransformer(num_classes=10)
        self.assertIsInstance(model, ProteinTransformer)

if __name__ == '__main__':
    unittest.main()