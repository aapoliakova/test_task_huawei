import os
from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive

URL = "train-clean-100"
_CHECKSUMS = {
    "http://www.openslr.org/resources/60/dev-clean.tar.gz": "da0864e1bd26debed35da8a869dd5c04dfc27682921936de7cff9c8a254dbe1a",
    # noqa: E501
    "http://www.openslr.org/resources/60/dev-other.tar.gz": "d413eda26f3a152ac7c9cf3658ef85504dfb1b625296e5fa83727f5186cca79c",
    # noqa: E501
    "http://www.openslr.org/resources/60/test-clean.tar.gz": "234ea5b25859102a87024a4b9b86641f5b5aaaf1197335c95090cde04fe9a4f5",
    # noqa: E501
    "http://www.openslr.org/resources/60/test-other.tar.gz": "33a5342094f3bba7ccc2e0500b9e72d558f72eb99328ac8debe1d9080402f10d",
    # noqa: E501
    "http://www.openslr.org/resources/60/train-clean-100.tar.gz": "c5608bf1ef74bb621935382b8399c5cdd51cd3ee47cec51f00f885a64c6c7f6b",
    # noqa: E501
    "http://www.openslr.org/resources/60/train-clean-360.tar.gz": "ce7cff44dcac46009d18379f37ef36551123a1dc4e5c8e4eb73ae57260de4886",
    # noqa: E501
    "http://www.openslr.org/resources/60/train-other-500.tar.gz": "e35f7e34deeb2e2bdfe4403d88c8fdd5fbf64865cae41f027a185a6965f0a5df",
    # noqa: E501
}


class LIBRITTS(Dataset):
    """Create a Dataset for *LibriTTS*.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"test-clean"``,
             ``"train-clean-100"``, ``"train-clean-360"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        maximum_sample_length (int):
            maximum_sample_length in seconds . (default: 2).
        sample_rate (int, optional):
            sample rate to resample audio wav. (default: 16_000 Hz).
                    download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    Returns: (tuple [torch.tensor, int])
                ``(waveform, label)``
                waveform shape [maximum_sample_length * sample_rate]

    """
    _ext_audio = ".wav"

    def __init__(self, root: Union[str, Path], url: str = URL,
                 folder_in_archive: str = "LibriTTS", download: bool = False,
                 maximum_sample_length: int = 2,
                 sample_rate: int = 16_000) -> None:

        self.sample_rate = sample_rate
        self.maximum_sample_length = maximum_sample_length * self.sample_rate
        self._speaker_genders = {}

        if url in [
            "dev-clean",
            "test-clean",
            "train-clean-100",
            "train-clean-360",
        ]:
            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/60/"
            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        basename = os.path.basename(url)
        archive = os.path.join(root, basename)
        speakers_path = os.path.join(root, folder_in_archive, "speakers.tsv")

        # Extracting labels to dict

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))

        with open(speakers_path, "r") as tsv_file:
            for line in tsv_file.readlines()[1:]:
                speaker_id, gender, _, _ = line.split("\t")
                self._speaker_genders[int(speaker_id)] = int(gender == "M")

    def __getitem__(self, n: int) -> tuple[Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, ):
            ``(waveform, label)``
        """
        fileid = self._walker[n]
        return self.load_libri_tts_item(fileid, self._path, self._ext_audio, self._speaker_genders)

    def load_libri_tts_item(self, fileid: str, path: str, ext_audio: str, labels: dict[int, int]) -> Tuple[Tensor, int]:
        speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
        utterance_id = fileid

        file_audio = utterance_id + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
        waveform = transform(waveform)
        return waveform[0, :self.maximum_sample_length], labels[int(speaker_id)]

    def __len__(self) -> int:
        return len(self._walker)


if __name__ == "__main__":
    dataset = LIBRITTS(root='data', download=True, sample_rate=8000, maximum_sample_length=2)
    wave_form, label = dataset[10000]
    print(f"Wave form shape: {wave_form.shape}")
    print(f"Label: {label}")


