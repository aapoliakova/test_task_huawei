import os
from typing import Dict, Any

import torch
import torchaudio
import wandb
from torch.utils.data import DataLoader
from datamodules.dataset import LIBRITTS
from utils.collate_fn import Collator
import torch.nn as nn

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LbriTTSDataModule(pl.LightningDataModule):
    """
    Args:
        root: (str or Path): Path to the directory where the dataset is found or downloaded.
        batch_size: (int) default 64
    """

    def __init__(self, root: str = 'data', batch_size: int = 64, **kwargs):
        super().__init__()
        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.batch_size = batch_size
        self.root = root
        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # download
        # LIBRITTS(root=self.root, download=True, url="train-clean-100")
        LIBRITTS(root=self.root, download=True, url="train-clean-360")
        LIBRITTS(root=self.root, download=True, url="dev-clean")
        LIBRITTS(root=self.root, download=True, url="test-clean")

    def setup(self, stage=None):
        # self.train_set = LIBRITTS(root=self.root, download=False, url="train-clean-100")
        self.train_set = LIBRITTS(root=self.root, download=False, url="train-clean-360")
        self.val_set = LIBRITTS(root=self.root, download=False, url="dev-clean")
        self.test_set = LIBRITTS(root=self.root, download=False, url="test-clean")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                           num_workers=os.cpu_count(), drop_last=True,
                                           pin_memory=True, collate_fn=Collator()
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=os.cpu_count(), drop_last=False,
                                           pin_memory=True, collate_fn=Collator()
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=os.cpu_count(), drop_last=False,
                                           pin_memory=True, collate_fn=Collator()
                                           )
