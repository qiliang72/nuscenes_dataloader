from torch.utils.data import DataLoader

from datasets import NuScenesDataset


def main() -> None:
    dataset = NuScenesDataset(
        data_root="data/nuscenes",
        split="train",
        sample_count=10,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    for batch_index, batch in enumerate(dataloader):
        print(f"Batch {batch_index}")
        print(batch)


if __name__ == "__main__":
    main()
