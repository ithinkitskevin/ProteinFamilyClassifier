import pytorch_lightning as pl


def train_model(model, train_data, val_data, epochs, batch_size, gpus=0):
    # Code for training the model
    trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)

    # TODO: Include callbacks for early stopping and checkpointing
    trainer.fit(model, train_data, val_data)
