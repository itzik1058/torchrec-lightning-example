from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class MovieLensDataset(Dataset):
    def __init__(
        self,
        path: Path,
        partition: str,
    ):
        self._data = pd.read_csv(
            path / partition,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        ).drop(["timestamp"], axis="columns")
        self._user_metadata = pd.read_csv(
            path / "u.user",
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            index_col="user_id",
            encoding="iso-8859-1",
        ).drop(["zip_code"], axis="columns")
        self._item_metadata = (
            pd.read_csv(
                path / "u.item",
                sep="|",
                names=[
                    "movie_id",
                    "movie_title",
                    "release_date",
                    "video_release_date",
                    "IMDb_URL",
                    "unknown",
                    "Action",
                    "Adventure",
                    "Animation",
                    "Children's",
                    "Comedy",
                    "Crime",
                    "Documentary",
                    "Drama",
                    "Fantasy",
                    "Film-Noir",
                    "Horror",
                    "Musical",
                    "Mystery",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "War",
                    "Western",
                ],
                index_col="movie_id",
                parse_dates=["release_date", "video_release_date"],
                encoding="iso-8859-1",
            )
            .rename({"movie_id": "item_id"}, axis="columns")
            .drop(
                ["movie_title", "video_release_date", "IMDb_URL"],
                axis="columns",
            )
        )
        self._genres = pd.read_csv(
            path / "u.genre",
            sep="|",
            names=["name", "id"],
            index_col="id",
        )
        for label in ["gender", "occupation"]:
            self._user_metadata[label] = LabelEncoder().fit_transform(
                self._user_metadata[label]
            )
        self._item_metadata["release_year"] = self._item_metadata[
            "release_date"
        ].dt.year.fillna(0)
        self._item_metadata = self._item_metadata.drop(["release_date"], axis="columns")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        user_id, item_id, rating = self._data.iloc[index].to_list()
        genres = (
            self._item_metadata.loc[item_id, self._genres["name"]]
            .to_numpy()
            .nonzero()[0]
        )
        sparse_values = [
            torch.tensor([user_id]),
            torch.tensor([item_id]),
            torch.tensor([self._user_metadata.loc[user_id, "gender"]]),
            torch.tensor([self._user_metadata.loc[user_id, "occupation"]]),
            torch.tensor(genres),
        ]
        sparse_lengths = torch.tensor([1, 1, 1, 1, len(genres)])
        dense_features = torch.tensor(
            [
                self._user_metadata.loc[user_id, "age"],
                self._item_metadata.loc[item_id, "release_year"],
            ],
            dtype=torch.float,
        )
        labels = torch.tensor(rating, dtype=torch.float)
        return {
            "sparse_values": sparse_values,
            "sparse_lengths": sparse_lengths,
            "dense_features": dense_features,
            "labels": labels,
        }


def movielens_collate_fn(batch):
    batch = dict(zip(batch[0].keys(), zip(*map(dict.values, batch))))
    return {
        "sparse_values": [torch.cat(v) for v in zip(*batch["sparse_values"])],
        "sparse_lengths": torch.stack(batch["sparse_lengths"], dim=1).flatten(),
        "dense_features": torch.stack(batch["dense_features"]),
        "labels": torch.stack(batch["labels"]),
    }


class MovieLensDataModule(LightningDataModule):
    def __init__(
        self,
        path: Path,
        train_partition: str,
        val_partition: str,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_dataset = MovieLensDataset(path, train_partition)
        self.val_dataset = MovieLensDataset(path, val_partition)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=movielens_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=movielens_collate_fn,
        )
