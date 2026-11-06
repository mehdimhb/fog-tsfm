from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from ..data import BaseData


class DataModule(LightningDataModule):
    def __init__(self, data: BaseData, batch_size: int, num_workers: int, shuffle: bool):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage: str) -> None:
        self.data.setup()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.data.collator,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.data.collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.data.collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def on_train_epoch_start(self):
        if self.data.train_ds:
            self.data.train_ds.set_epoch(self.trainer.current_epoch)
