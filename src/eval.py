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


def inference(config):
    seed_everything(seed=42)

    wandb_logger = WandbLogger(project="gender_classification",
                               log_model=True, config=config)
    config = wandb.config
    data_model = LbriTTSDataModule(**config)
    cls_model = AudioClassifier(**config)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=10,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         gradient_clip_val=5)
    trainer.test(datamodule=data_model, model=cls_model, ckpt_path=config['ckpt'])
    wandb.finish()


if __name__ == "__main__":
    argument_parser = ArgumentParser(description="Tran gender classifier by voice")
    argument_parser.add_argument("--batch_size", type=int, default=128)
    argument_parser.add_argument("--dropout", type=float, default=0.2)
    argument_parser.add_argument("--hidden_size", type=int, default=40)
    argument_parser.add_argument("--lr", type=float, default=1e-3)
    argument_parser.add_argument("--n_mels", type=int, default=40)
    argument_parser.add_argument("--num_layers", type=int, default=2)
    argument_parser.add_argument("--ckpt", type=str)
    argument_parser.add_argument("--encoder_type", type=str, default='rnn')

    args = argument_parser.parse_args()
    config_ = vars(args)
    inference(config_)
