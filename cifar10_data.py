from __future__ import annotations

import pickle
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_DIR = "cifar-10-batches-py"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _download_file(url: str, target: Path) -> None:
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    if tmp_target.exists():
        tmp_target.unlink()

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        subprocess.run(
            [
                curl,
                "-L",
                "--fail",
                "--connect-timeout",
                "20",
                "--max-time",
                "900",
                "--retry",
                "2",
                "-o",
                str(tmp_target),
                url,
            ],
            check=True,
        )
    else:
        with urllib.request.urlopen(url, timeout=20) as response:
            with tmp_target.open("wb") as file:
                shutil.copyfileobj(response, file)

    if tmp_target.stat().st_size == 0:
        raise OSError(f"Downloaded empty file from {url}")
    shutil.move(str(tmp_target), target)


def download_cifar10(data_dir: str | Path) -> Path:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    extracted = data_path / CIFAR10_DIR
    if extracted.exists():
        return extracted

    archive = data_path / CIFAR10_ARCHIVE
    if not archive.exists():
        print(f"Downloading CIFAR-10 to {archive} ...", flush=True)
        _download_file(CIFAR10_URL, archive)

    print(f"Extracting {archive} ...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_path)
    return extracted


def _load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    with path.open("rb") as file:
        batch = pickle.load(file, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32)
    labels = batch["labels"]
    return images, labels


class CIFAR10Dataset(Dataset):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010)).view(3, 1, 1)

    def __init__(
        self,
        data_dir: str | Path = "data",
        train: bool = True,
        augment: bool = True,
        limit: int | None = None,
    ) -> None:
        root = download_cifar10(data_dir)
        self.train = train
        self.augment = augment and train

        if train:
            batch_paths = [root / f"data_batch_{index}" for index in range(1, 6)]
        else:
            batch_paths = [root / "test_batch"]

        images = []
        labels = []
        for batch_path in batch_paths:
            batch_images, batch_labels = _load_batch(batch_path)
            images.append(batch_images)
            labels.extend(batch_labels)

        image_array = np.concatenate(images, axis=0).astype(np.float32) / 255.0
        self.images = torch.from_numpy(image_array)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if limit is not None:
            self.images = self.images[:limit]
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment:
            image = random_crop_32(image, padding=4)
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=(2,))
        image = (image - self.mean) / self.std
        return image, self.labels[index]


def random_crop_32(image: torch.Tensor, padding: int = 4) -> torch.Tensor:
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
    padded = padded.squeeze(0)
    top = torch.randint(0, padding * 2 + 1, ()).item()
    left = torch.randint(0, padding * 2 + 1, ()).item()
    return padded[:, top : top + 32, left : left + 32]
