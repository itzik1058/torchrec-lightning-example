from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.models.dlrm import DLRM


class LightningDLRM(LightningModule):
    def __init__(
        self,
        keys: list[str],
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: list[int],
        over_arch_layer_sizes: list[int],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.model: DLRM = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
        )
        self.loss_fn: nn.Module = nn.MSELoss()

    def forward(
        self,
        sparse_values: torch.Tensor,
        sparse_lengths: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            dense_features,
            KeyedJaggedTensor(
                keys=self.keys,
                values=sparse_values,
                lengths=sparse_lengths,
            ),
        )
        return output.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def _step(self, batch, batch_idx: int, step_phase: str):
        logits = self.forward(
            torch.cat(batch["sparse_values"]),
            batch["sparse_lengths"],
            batch["dense_features"],
        )
        loss = self.loss_fn(logits, batch["labels"].squeeze().float())

        self.log(f"{step_phase}_loss", loss, batch_size=batch["labels"].size(0))
        return loss

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "test")
