def train_model(model, train_data, val_data, epochs, batch_size):
    # Code for training the model
    model.fit(train_data, epochs=epochs, batch_size=batch_size)
    # Save model, logging, etc.