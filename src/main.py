import argparse
from data_preprocessing import load_data, preprocess_data
from model import create_model
from train import train_model
from evaluate import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Classifier Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    # Add more arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()
    data = load_data('data/path')
    processed_data = preprocess_data(data)
    model = create_model(input_shape, num_classes)
    train_model(model, processed_data, args.epochs, args.batch_size)
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
