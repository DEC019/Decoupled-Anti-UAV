# Decoupled-Anti-UAV

Code for **Rethinking Proposal Generation and UAV Recognition for Generalizable Anti-UAV Detection in Unseen Scenes**.

This project implements a proposal–recognition decoupled framework for detecting small UAVs in unseen scenes. A general-purpose detector generates high-recall object proposals, and a dedicated image classifier determines which proposals contain UAVs. Separating these tasks allows localization and recognition to benefit from complementary data sources, reducing reliance on scarce bounding-box annotations for small UAVs.

## Environment Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository:

```bash
git clone https://github.com/DEC019/Decoupled-Anti-UAV.git
cd Decoupled-Anti-UAV
uv sync --python 3.11
```

Run these commands from the directory containing [pyproject.toml](./pyproject.toml) and [detect_yolo.py](./detect_yolo.py). If you already have the source code locally, change to that directory and run `uv sync --python 3.11` directly.

`uv sync` creates a local `.venv` and installs the project dependencies. Use `uv run` for subsequent commands; manual environment activation is unnecessary. Python 3.11 is recommended for the pinned dependency versions. Key dependencies include:

| Package | Version |
| --- | --- |
| Ultralytics | 8.4.62 |
| PyTorch | 2.5.1 |
| Torchvision | 0.20.1 |

The complete dependency list is defined in [pyproject.toml](./pyproject.toml). GPU inference requires an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch installation. Use `--device cpu` to run on the CPU.
