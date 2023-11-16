import argparse
from model import create_model
from train import train_model
from evaluate import evaluate_model
from dataset import ProteinSequenceDataset, reader, build_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Protein Classifier Training")

    parser.add_argument(
        "--type",
        type=str,
        default="train",
        choices=["train", "evaluate"],
        help="Type of run",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Path to model directory"
    )

    parser.add_argument(
        "--seq_max_len", type=int, default=120, help="Length of protein sequence"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/random_split",
        help="Path to data directory",
    )
    parser.add_argument(
        "--data_workers", type=int, default=2, help="Number of workers for data loading"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )

    return parser.parse_args()


def train(args):
    train_data, train_targets = reader("train", data_dir)
    dev_data, dev_targets = reader("dev", data_dir)

    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    train_dataset = SequenceDataset(
        word2id, fam2label, seq_max_len, train_data, train_targets
    )
    dev_dataset = SequenceDataset(
        word2id, fam2label, seq_max_len, dev_data, dev_targets
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.data_workers,
    )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_workers,
    )

    model = create_model(input_shape, num_classes)
    train_model(model, processed_data, args.epochs, args.batch_size)


def evaluate(args):
    train_data, train_targets = reader("train", data_dir)
    test_data, test_targets = reader("test", data_dir)

    word2id = build_vocab(train_data)
    fam2label = build_labels(train_targets)

    test_dataset = SequenceDataset(
        word2id, fam2label, seq_max_len, test_data, test_targets
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model_path = args.model_path
    model = pickle.load(open(model_path, "rb"))
    evaluate_model(model, test_data)


def main():
    args = parse_args()

    if lower(args.type) == "train":
        train(args)
    elif lower(args.type) == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
