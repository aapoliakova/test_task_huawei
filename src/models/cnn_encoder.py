import torch.nn as nn
import torch
import pytorch_lightning as pl


class MaxOverTime(pl.LightningModule):
    def __init__(self):
        super(MaxOverTime, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return x.max(2)[0]


class CNNEncoder(pl.LightningModule):
    def __init__(self, input_size, hidden_size, dropout, *args, **kwargs):
        super(CNNEncoder, self).__init__()
        self.conv10 = nn.Sequential(
            *[nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=10, stride=5),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(), MaxOverTime()])
        self.conv20 = nn.Sequential(
            *[nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=20, stride=10),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(), MaxOverTime()])
        self.conv40 = nn.Sequential(
            *[nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=40, stride=20),
              nn.BatchNorm1d(hidden_size),
              nn.ReLU(), MaxOverTime()])
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(hidden_size * 3, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        out10 = self.conv10(batch['x'])
        out20 = self.conv20(batch['x'])
        out40 = self.conv40(batch['x'])
        out = torch.cat([out10, out20, out40], -1)
        out = self.logits(out)
        return out
