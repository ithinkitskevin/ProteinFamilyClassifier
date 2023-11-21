"""
Module for evaluating the model using given data
"""

import pytorch_lightning as pl


def evaluate_model(model, data, args={}):
    """
    Evaluates a given model using a test data.

    Args:
        model (pl.LightningModule): The model to be evaluated.
        data (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        args (dict, optional): Additional arguments for PyTorch Lightning Trainer. Defaults to an empty dict.

    Returns:
        list: A list containing the results of the test evaluation. The content of the list
              depends on the metrics and settings used in the model's test_step and test_epoch_end methods.
    """
    model.eval()

    pl.seed_everything(0)
    trainer = pl.Trainer()

    result = trainer.test(model=model, dataloaders=data)

    return result