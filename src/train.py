import gc
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from datamodules.datamodule import LbriTTSDataModule
from models.base_classifier import AudioClassifier
import wandb
import torch
from argparse import ArgumentParser
import os

os.environ["WANDB_DISABLED"] = "true"


def train(config=None):
    seed_everything(seed=42)

    name = f"{config['encoder_type']}: {config['lr']}, N: {config['batch_size']}, "
    wandb_logger = WandbLogger(project="gender_classification360", name=name,
                               log_model=True, config=config)
    config = wandb.config
    data_model = LbriTTSDataModule(**config)
    cls_model = AudioClassifier(**config)

    checkpoint_callback = ModelCheckpoint(monitor="validation loss",
                                          mode="min", save_top_k=1)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=7,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         val_check_interval=500,
                         gradient_clip_val=5,
                         callbacks=[
                             checkpoint_callback
                         ]
                         )
    trainer.fit(model=cls_model, datamodule=data_model)
    trainer.test(cls_model, data_model, 'best')
    wandb.finish()


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Tran gender classifier by voice")
    argument_parser.add_argument("--batch_size", type=int, default=32)
    argument_parser.add_argument("--dropout", type=float, default=0.2)
    argument_parser.add_argument("--hidden_size", type=int, default=40)
    argument_parser.add_argument("--lr", type=float, default=1e-3)
    argument_parser.add_argument("--n_mels", type=int, default=40)
    argument_parser.add_argument("--num_layers", type=int, default=2)
    argument_parser.add_argument("--encoder_type", type=str, default='rnn')

    args = argument_parser.parse_args()
    config_ = vars(args)
    train(config_)
