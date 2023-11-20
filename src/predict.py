import pytorch_lightning as pl
import os
import pandas as pd
import torch

def read_predict_file(file_path):
    # Code for predicting a single file
    data = []
    with open(file_path) as file:
        data.append(
            pd.read_csv(
                file, index_col=None, usecols=["sequence"]
            )
        )

    all_data = pd.concat(data)
    return all_data["sequence"]

def predict_model(model, data, labels, args={}):
    labels = dict((v,k) for k,v in labels.items())

    model.eval()
    
    # Code for training the model
    pl.seed_everything(0)
    trainer = pl.Trainer()

    predictions = trainer.predict(model=model, dataloaders=data)
    softmaxed_outputs = [torch.softmax(tensor, dim=1) for tensor in predictions]

    # Then, use argmax to get the predicted classes for each tensor
    predicted_classes = []
    for tensor in softmaxed_outputs:
        predicted_classes.extend(torch.argmax(tensor, dim=1).tolist())
    
    final_predictions = []
    # Print the predicted classes
    for pred in predicted_classes:
        final_predictions.append(labels.get(pred, "unk"))
        
    return final_predictions