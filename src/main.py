"""
This script provides functionality for training, evaluating, and predicting protein
classifications using convolutional neural networks (CNN)
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import SequenceDataset, build_labels, build_vocab, reader
from evaluate import evaluate_model
from model import ProtCNN
from predict import predict_model, read_predict_file
from train import train_model


def parse_args():
    """
    Parses command line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Protein Classifier Training")

    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="Type of run",
    )
    parser.add_argument(
        "--model_type", type=str, default="cnn", choices=["cnn"]
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Path to model directory"
    )

    parser.add_argument(
        "--seq_max_len", type=int, default=120, help="Length of protein sequence"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/random_split",
        help="Path to data directory",
    )
    parser.add_argument(
        "--predict_dir",
        type=str,
        default="data/raw/sample/test",
        help="Path to predict directory",
    )

    parser.add_argument(
        "--data_workers", type=int, default=1, help="Number of workers for data loading"
    )

    return parser.parse_args()


def train(args):
    """
    Trains a protein classification model and saves it.

    Args:
        args (argparse.Namespace): Command line arguments with training parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_targets = reader("train", args.data_dir)
    dev_data, dev_targets = reader("dev", args.data_dir)

    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    train_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, train_data, train_targets
    )
    dev_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, dev_data, dev_targets
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_workers,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
    )

    num_classes = len(fam2label)

    model = None
    if args.model_type == "cnn":
        model = ProtCNN(num_classes)
    else:
        print("Invalid model type")
        return 0
    model.to(device)

    train_model(model, train_dataloader, dev_dataloader, args)


def evaluate(args):
    """
    Evaluates a trained protein classification model.

    Args:
        args (argparse.Namespace): Command line arguments with evaluation parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, train_targets = reader("train", args.data_dir)

    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    test_data, test_targets = reader("test", args.data_dir)

    test_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, test_data, test_targets
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
    )
    num_classes = len(fam2label)

    model = None
    if args.model_type == "cnn":
        model = ProtCNN.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    else:
        print("Invalid model type")
        return 0
    model.to(device)

    evaluate_model(model, test_dataloader, args)


def predict(args):
    """
    Makes predictions using a trained protein classification model.

    Args:
        args (argparse.Namespace): Command line arguments with prediction parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load training data for building vocab and labels (as before)
    train_data, train_targets = reader("train", args.data_dir)

    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)
    num_classes = len(fam2label)

    # Load prediction data
    predict_data = read_predict_file(args.predict_dir)

    # Create a dataset for prediction. Make sure this dataset returns only sequence tensors
    predict_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, predict_data, None
    )

    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
    )

    model = None
    if args.model_type == "cnn":
        model = ProtCNN.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    else:
        print("Invalid model type")
        return 0
    model.to(device)
    print("Loaded model")

    final_predictions = predict_model(model, predict_dataloader, fam2label, args)

    lines = []
    with open(args.predict_dir, "r") as file:
        lines = file.readlines()

    # Add 'prediction' to the header
    header = lines[0].strip() + ",prediction\n"
    updated_lines = [header]

    # Append predictions to each line
    for line, prediction in zip(lines[1:], final_predictions):
        updated_line = line.strip() + "," + str(prediction) + "\n"
        updated_lines.append(updated_line)

    # Write the updated lines to a new file
    with open(args.predict_dir + "_prediction", "w") as file:
        file.writelines(updated_lines)

    print("Finished writing predictions to file")


def main():
    """
    Main function to run the training, evaluation, or prediction.
    """
    # For my machine, utilizes GPU for training
    torch.set_float32_matmul_precision("high")

    args = parse_args()

    if args.type.lower() == "train":
        train(args)
    elif args.type.lower() == "evaluate":
        evaluate(args)
    elif args.type.lower() == "predict":
        predict(args)
    else:
        print("Invalid run type")


if __name__ == "__main__":
    main()
