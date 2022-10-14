import os
from typing import Dict, Any

import torch
import torchaudio
import wandb

import torch.nn as nn

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class Collator:
    def __call__(self, batch: list[tuple[torch.tensor, int]]):
        waveforms, labels = zip(*batch)
        waveforms_batch = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        labels = torch.tensor(labels).long()
        length = torch.tensor([wav.size(-1) for wav in waveforms]).long()
        return {'x': waveforms_batch,
                'labels': torch.unsqueeze(labels, 1),
                'mask': length}
