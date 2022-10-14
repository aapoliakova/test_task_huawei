import torch
from typing import Optional
import torch.nn as nn
import torchaudio


class FeatureExtractor(nn.Module):
    """Extract mel spectrogram and recalculate sample length.

    Args:
        n_mels (int): number of frequencies
    Returns:
        spectrogram of shape [bs, n_mels, seq_len - win_len / hop_len]
        seq_length of shape: [bs]
    """

    def __init__(self, n_mels: int = 80):
        super().__init__()
        self._feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=n_mels,
            center=True)

    def __call__(self, wav: torch.tensor, seq_length: torch.tensor = None):
        """
        Args:
            wav: batch of samples, shape [bs, sample length]
            seq_length: sample length [bs] or None if mask not used for training
        Returns:
            dict

        """
        self._mel_spectrogram = self._feature_extractor(wav)
        self._mel_spectrogram = self._mel_spectrogram.clamp(min=1e-5).log()

        if seq_length is not None:
            seq_length = (seq_length - self._feature_extractor.win_length) // self._feature_extractor.hop_length
            seq_length += 4
            return {'x': self._mel_spectrogram,
                    'mask': seq_length}
        return {'x': self._mel_spectrogram}


class Collator:
    def __call__(self, batch: list[tuple[torch.tensor, int]]):
        waveforms, labels = zip(*batch)
        waveforms_batch = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        labels = torch.tensor(labels).long()
        length = torch.tensor([wav.size(-1) for wav in waveforms]).long()
        return {'x': waveforms_batch,
                'labels': torch.unsqueeze(labels, 1),
                'mask': length}


if __name__ == "__main__":
    fc = FeatureExtractor()
