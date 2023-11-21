"""
Module for handling dataset operations for protein sequence classification.
"""

import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Define the early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",  # The metric to monitor
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,  # Whether to print logs to stdout
    mode="min",  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
)


def save_model_checkpoint(trainer, args):
    """
    Saves a model checkpoint.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning Trainer instance used for training.
        args (Namespace): Command line arguments containing model type and other parameters.
    """
    # Get the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Format the checkpoint filename
    checkpoint_filename = f"{args.model_type}_{timestamp}.ckpt"

    # Combine the directory and filename
    checkpoint_path = os.path.join("models", checkpoint_filename)

    # Save the model checkpoint
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train_model(model, train_data, val_data, args):
    """
    Trains the model using the provided training and validation data.

    Args:
        model (pl.LightningModule): The model to be trained.
        train_data (DataLoader): DataLoader for the training dataset.
        val_data (DataLoader): DataLoader for the validation dataset.
        args (Namespace): Command line arguments containing training parameters such as epochs.
    """
    # Code for training the model
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=[early_stop_callback])

    trainer.fit(model, train_data, val_data)

    # Save the trained model checkpoint
    save_model_checkpoint(trainer, args)