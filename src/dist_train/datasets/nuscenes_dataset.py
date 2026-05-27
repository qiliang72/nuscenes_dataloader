from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class NuScenesDataset(Dataset[dict[str, Any]]):
    """A minimal nuScenes-style dataset used to learn PyTorch data loading."""

    def __init__(self, data_root: str, split: str = "train", sample_count: int = 10) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be greater than or equal to 0.")

        self.data_root = Path(data_root)
        self.split = split
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.sample_count:
            raise IndexError(f"Sample index {index} is out of range.")

        return {
            "sample_token": f"{self.split}-sample-{index:06d}",
            "scene_name": f"scene-{index // 10:04d}",
            "timestamp": index * 100_000,
            "split": self.split,
        }
