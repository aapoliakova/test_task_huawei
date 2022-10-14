import os
from typing import Dict, Any

import torch
import torchaudio

import torch.nn as nn
from datamodules.feature_extractor import FeatureExtractor
from models.rnn_encoder import RNNEncoder
from models.cnn_encoder import CNNEncoder
import torch

import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy


class AudioClassifier(pl.LightningModule):
    def __init__(self, n_mels: int = 80,
                 hidden_size: int = 20,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 lr: float = 1e-3,
                 encoder_type: str = "rnn",
                 **kwargs):
        super(AudioClassifier, self).__init__()

        self.save_hyperparameters()
        # Preprocessing hyperparameters
        self.transform = FeatureExtractor(n_mels=n_mels)

        if encoder_type == "rnn":
            self.encoder = RNNEncoder(input_size=n_mels, hidden_size=hidden_size, num_layers=num_layers,
                                      dropout=dropout)
        elif encoder_type == "cnn":
            self.encoder = CNNEncoder(input_size=n_mels, hidden_size=hidden_size, num_layers=num_layers,
                                      dropout=dropout)
        # Optimization hyperparameters
        self.metrics = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr

    def prepare_batch(self, batch):
        out = self.transform(batch['x'], batch['mask'])
        batch['x'], batch['mask'] = out['x'], out['mask']
        return batch

    def forward(self, batch):
        batch = self.prepare_batch(batch)
        logits = self.encoder(batch)
        return logits

    def run_step(self, batch, stage):
        logits = self.forward(batch)
        loss = self.loss(logits, batch['labels'].float())
        acc = self.metrics(logits, batch['labels'])
        self.log(f"{stage} loss", loss, on_step=False, on_epoch=True)
        self.log(f"{stage} accuracy", acc, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch):
        return self.run_step(batch, "training")

    def validation_step(self, batch, *args, **kwargs):
        return self.run_step(batch, "validation")

    def test_step(self, batch, batch_idx):
        return self.run_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation loss"}
