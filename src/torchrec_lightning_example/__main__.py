from pathlib import Path

import onnx
import onnxruntime
import torch
from lightning.pytorch.cli import LightningCLI
from torch.export.dynamic_shapes import Dim

from torchrec_lightning_example.data import MovieLensDataModule
from torchrec_lightning_example.model import LightningDLRM


def main():
    cli = LightningCLI(LightningDLRM, MovieLensDataModule)

    # onnx conversion fails for padding_idx None
    @torch.no_grad()
    def embedding_bag_padding_zero(m):
        if type(m) is torch.nn.EmbeddingBag:
            m.padding_idx = 0

    cli.model.apply(embedding_bag_padding_zero)

    sample = next(iter(cli.datamodule.train_dataloader()))
    onnx_path = Path("env/model.onnx")
    cli.model.to_onnx(
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_sample=(
            torch.cat(sample["sparse_values"]),
            sample["sparse_lengths"],
            sample["dense_features"],
        ),
        input_names=["sparse_values", "sparse_lengths", "dense_features"],
        output_names=["output"],
        dynamic_shapes={
            "sparse_values": (Dim.AUTO,),  # type: ignore
            "sparse_lengths": (Dim.AUTO,),  # type: ignore
            "dense_features": (Dim.AUTO, Dim.AUTO),  # type: ignore
        },
        dynamo=True,
    )
    onnx.checker.check_model(onnx.load(onnx_path))

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {
        "sparse_values": torch.cat(sample["sparse_values"]).numpy(),
        "sparse_lengths": sample["sparse_lengths"].numpy(),
        "dense_features": sample["dense_features"].numpy(),
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    print(
        "onnx conversion mse",
        torch.nn.functional.mse_loss(
            torch.tensor(ort_outputs[0]),
            cli.model(
                torch.cat(sample["sparse_values"]),
                sample["sparse_lengths"],
                sample["dense_features"],
            ),
        ).item(),
    )


if __name__ == "__main__":
    main()
