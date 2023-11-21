"""
Module for predicting the model using given data
Also reads the data given in the path
"""

import pandas as pd
import pytorch_lightning as pl
import torch

def read_predict_file(file_path):
    """
    Reads a CSV file containing protein sequences for prediction.

    Args:
        file_path (str): Path to the CSV file containing protein sequences.

    Returns:
        pandas.Series: A series containing the protein sequences.
    """
    # Read data from the CSV file
    data = []
    with open(file_path) as file:
        data.append(pd.read_csv(file, index_col=None, usecols=["sequence"]))

    # Concatenate all data into a single pandas Series
    all_data = pd.concat(data)
    return all_data["sequence"]


def predict_model(model, data, labels, args={}):
    """
    Makes predictions using a trained model on the provided data.

    Args:
        model (pl.LightningModule): The trained model for making predictions.
        data (DataLoader): DataLoader containing the data for prediction.
        labels (dict): A dictionary mapping class indices to class labels.
        args (dict, optional): Additional arguments for PyTorch Lightning Trainer. Defaults to an empty dict.

    Returns:
        list: A list of predicted class labels for each input sequence.
    """
    # Reverse the labels dictionary for easier lookup
    labels = dict((v, k) for k, v in labels.items())

    # Set model to evaluation mode and ensure reproducibility
    model.eval()
    pl.seed_everything(0)

    # Initialize a PyTorch Lightning Trainer for prediction
    trainer = pl.Trainer()

    # Perform predictions using the trained model
    predictions = trainer.predict(model=model, dataloaders=data)

    # Apply softmax to convert logits to probabilities and use argmax to find predicted classes
    softmaxed_outputs = [torch.softmax(tensor, dim=1) for tensor in predictions]
    predicted_classes = [torch.argmax(tensor, dim=1).tolist() for tensor in softmaxed_outputs]
    
    # Flatten the list of predicted classes
    predicted_classes_flat = [item for sublist in predicted_classes for item in sublist]

    # Convert predicted class indices to labels
    final_predictions = [labels.get(pred, "unk") for pred in predicted_classes_flat]

    return final_predictions
