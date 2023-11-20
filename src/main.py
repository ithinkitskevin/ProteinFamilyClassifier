import argparse
import torch
from torch.utils.data import DataLoader
from model import ProtCNN, ProteinTransformer
from train import train_model
from evaluate import evaluate_model
from predict import predict_model, read_predict_file
from dataset import SequenceDataset, reader, build_labels, build_vocab
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Classifier Training")

    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "evaluate","predict"],
        help="Type of run",
    )
    parser.add_argument(
        "--model_type", type=str, default="cnn", choices=["cnn", "transformer"]
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
        help="Path to data directory",
    )
    
    parser.add_argument(
        "--data_workers", type=int, default=3, help="Number of workers for data loading"
    )

    return parser.parse_args()


def train(args):
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
    if args.model_type == "transformer":
        model = ProteinTransformer(num_classes)
    elif args.model_type == "cnn":
        model = ProtCNN(num_classes)
    else:
        print("Invalid model type")
        return 0
    model.cuda()
    
    train_model(model, train_dataloader, dev_dataloader, args)
    

def evaluate(args):
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
    if args.model_type == "transformer":
        model = ProteinTransformer.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    elif args.model_type == "cnn":
        model = ProtCNN.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    else:
        print("Invalid model type")
        return 0
    print("Loaded model")
    
    evaluate_model(model, test_dataloader, args)


def predict(args):
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
    if args.model_type == "transformer":
        model = ProteinTransformer.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    elif args.model_type == "cnn":
        model = ProtCNN.load_from_checkpoint(args.model_dir, num_classes=num_classes)
    else:
        print("Invalid model type")
        return 0
    print("Loaded model")
    
    final_predictions = predict_model(model, predict_dataloader, fam2label, args)lines = []
    
    with open(args.predict_dir, 'r') as file:
        lines = file.readlines()

    # Add 'prediction' to the header
    header = lines[0].strip() + ',prediction\n'
    updated_lines = [header]

    # Append predictions to each line
    for line, prediction in zip(lines[1:], final_predictions):
        updated_line = line.strip() + ',' + str(prediction) + '\n'
        updated_lines.append(updated_line)

    # Write the updated lines to a new file
    with open(args.predict_dir+"_prediction", 'w') as file:
        file.writelines(updated_lines)

    print("Finished writing predictions to file")

def main():
    torch.set_float32_matmul_precision('high')

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
