"""Config loading. Every stage reads its parameters from configs/default.yaml."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class Paths:
    raw_dir: Path
    processed_dir: Path
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Paths":
        p = cfg["paths"]
        paths = cls(
            raw_dir=Path(p["raw_dir"]),
            processed_dir=Path(p["processed_dir"]),
            output_dir=Path(p["output_dir"]),
        )
        for d in (paths.raw_dir, paths.processed_dir, paths.output_dir):
            d.mkdir(parents=True, exist_ok=True)
        return paths
