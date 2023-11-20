import pytorch_lightning as pl


def evaluate_model(model, test_dataloader, args={}):
    model.eval()
    
    pl.seed_everything(0)
    trainer = pl.Trainer()

    result = trainer.test(model=model, dataloaders=test_dataloader)
    
    return result