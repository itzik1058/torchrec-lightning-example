# torchrec-lightning-example

This is a simple example of using TorchRec with Pytorch Lightning on the MovieLens dataset. The goal is to build a recommendation system that predicts user ratings for movies.

## Installation

```bash
pip install git+https://github.com/itzik1058/torchrec-lightning-example
```

## Usage

```bash
torchrec-lightning-example fit -c config/ml-100k.yaml
```

Model checkpoints are saved in the `env` directory.

## Optional

To install only CPU dependencies with uv use pytorch-cpu index

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
torchrec = [
  { index = "pytorch-cpu" },
]
fbgemm-gpu = [
  { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```
