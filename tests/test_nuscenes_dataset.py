from torch.utils.data import DataLoader

from dist_train.datasets import NuScenesDataset


def test_dataset_length_matches_sample_count() -> None:
    dataset = NuScenesDataset(data_root="data/nuscenes", sample_count=3)

    assert len(dataset) == 3


def test_dataset_item_contains_expected_fields() -> None:
    dataset = NuScenesDataset(data_root="data/nuscenes", split="mini_train", sample_count=1)

    sample = dataset[0]

    assert sample == {
        "sample_token": "mini_train-sample-000000",
        "scene_name": "scene-0000",
        "timestamp": 0,
        "split": "mini_train",
    }


def test_dataloader_can_iterate_one_batch() -> None:
    dataset = NuScenesDataset(data_root="data/nuscenes", sample_count=4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    assert batch["sample_token"] == ["train-sample-000000", "train-sample-000001"]
    assert batch["scene_name"] == ["scene-0000", "scene-0000"]
    assert batch["timestamp"].tolist() == [0, 100_000]
    assert batch["split"] == ["train", "train"]
