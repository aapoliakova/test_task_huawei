import torch.nn as nn
import torch
import pytorch_lightning as pl


class RNNEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=num_layers,
                          dropout=dropout)

        self.bn = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, batch):
        x, length = batch['x'], batch['mask']
        out, _ = self.gru(torch.permute(x, (0, 2, 1)))
        out = out[torch.arange(x.shape[0]), length, :]
        out = self.activation(self.bn(out))
        out = self.classifier(out)
        return out
