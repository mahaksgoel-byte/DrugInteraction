#!/usr/bin/env python
"""End-to-end DDI Prediction Pipeline runner.

Executes every stage in DDI_Pipeline_Architecture.html in order:
  1. Data sources -> 2. ID harmonization -> Data cleaning -> Class balance
  -> Train/val/test split (with cold-start) -> 3-4. Molecular parsing &
  heterogeneous graph build -> 5. Node features -> 6. Self-supervised
  pretraining -> 7. R-GCN encoder fine-tuning -> 8. Drug-pair pooling ->
  9. Bottleneck projection -> Classical head vs. Quantum head (Qiskit) ->
  10. Evaluation & comparison.

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddi_pipeline.config import Paths, load_config, set_seed
from ddi_pipeline.data_sources import load_all_sources
from ddi_pipeline.evaluation import compare_heads
from ddi_pipeline.gnn_backbone import DDIBackbone, pretrain_encoder
from ddi_pipeline.graph_construction import build_hetero_graph
from ddi_pipeline.preprocessing import class_balance_check, clean_data, harmonize_ids, train_val_test_split
from ddi_pipeline.training import (
    build_classical_head,
    build_quantum_head,
    compute_bottleneck_features,
    train_head,
    train_head_on_features,
)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    paths = Paths.from_config(cfg)
    device = "cuda" if _cuda_available() else "cpu"
    print(f"[run_pipeline] device={device}")

    # ---- Stage 1: Data sources ----
    sources = load_all_sources(cfg, paths.raw_dir)

    # ---- Stage 2: ID harmonization ----
    sources = harmonize_ids(sources, min_sources_required=cfg["preprocessing"]["min_sources_required"])

    # ---- Data cleaning ----
    sources = clean_data(sources)

    # ---- Class balance check ----
    class_balance_check(sources["drugbank_ddi"])

    # ---- Train/val/test split incl. cold-start ----
    split = train_val_test_split(
        sources["drugbank_ddi"],
        test_size=cfg["preprocessing"]["test_size"],
        val_size=cfg["preprocessing"]["val_size"],
        cold_start_fraction=cfg["preprocessing"]["cold_start_fraction"],
        seed=cfg["seed"],
    )

    # ---- Stages 3-5: molecular parsing, heterogeneous graph, node features ----
    data, drug_idx, protein_idx = build_hetero_graph(sources, cfg)

    # ---- Backbone: encoder + pooling + bottleneck (Stages 7-9) ----
    in_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    backbone = DDIBackbone(data.metadata(), in_dims, cfg)

    # ---- Stage 6: self-supervised pretraining ----
    gcfg = cfg["gnn"]
    backbone.encoder = pretrain_encoder(
        backbone.encoder, data, gcfg["pretrain_epochs"], gcfg["pretrain_lr"], gcfg["mask_rate"],
        final_hidden_dim=gcfg["hidden_dims"][-1], device=device,
    )

    # ---- Fine-tune the shared backbone once (with the classical head as a
    # supervised probe) so the bottleneck actually encodes DDI-relevant
    # signal before both heads train on frozen copies of it. ----
    print("\n[run_pipeline] Fine-tuning shared backbone (probe: classical head)...")
    probe_head = build_classical_head(cfg)
    train_head(
        backbone,
        probe_head,
        data,
        drug_idx,
        split.train,
        split.val,
        epochs=cfg["classical_head"]["epochs"],
        lr=cfg["classical_head"]["lr"],
        batch_size=cfg["classical_head"]["batch_size"],
        device=device,
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    # ---- Freeze backbone; materialize identical bottleneck features for
    # both heads (the diagram's "Fork point: identical 8-dim input to both
    # heads ensures a fair comparison"). ----
    for p in backbone.parameters():
        p.requires_grad_(False)

    train_x, train_y = compute_bottleneck_features(backbone, data, drug_idx, split.train, device)
    val_x, val_y = compute_bottleneck_features(backbone, data, drug_idx, split.val, device)
    test_x, test_y = compute_bottleneck_features(backbone, data, drug_idx, split.test, device)
    cold_x, cold_y = (
        compute_bottleneck_features(backbone, data, drug_idx, split.cold_start_test, device)
        if len(split.cold_start_test)
        else (train_x[:0], train_y[:0])
    )
    backbone_num_params = sum(p.numel() for p in backbone.parameters())

    # ---- Pipeline A: classical head ----
    print("\n[run_pipeline] Training classical head on frozen bottleneck...")
    classical_head = build_classical_head(cfg)
    classical_result = train_head_on_features(
        classical_head, train_x, train_y, val_x, val_y,
        epochs=cfg["classical_head"]["epochs"], lr=cfg["classical_head"]["lr"],
        batch_size=cfg["classical_head"]["batch_size"], device=device,
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    # ---- Pipeline B: quantum head ----
    print("\n[run_pipeline] Training quantum head (Qiskit simulator) on frozen bottleneck...")
    quantum_head = build_quantum_head(cfg)
    quantum_result = train_head_on_features(
        quantum_head, train_x, train_y, val_x, val_y,
        epochs=cfg["quantum_head"]["epochs"], lr=cfg["quantum_head"]["lr"],
        batch_size=cfg["quantum_head"]["batch_size"], device=device,
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    # ---- Stage 10: Evaluation & comparison ----
    print("\n[run_pipeline] Evaluating classical vs. quantum heads...")
    comparison = compare_heads(
        backbone_num_params=backbone_num_params,
        heads={"classical": classical_head, "quantum": quantum_head},
        test_features={"classical": (test_x, test_y), "quantum": (test_x, test_y)},
        cold_start_features={"classical": (cold_x, cold_y), "quantum": (cold_x, cold_y)},
        train_results={"classical": classical_result, "quantum": quantum_result},
    )

    out_csv = paths.output_dir / "comparison_results.csv"
    comparison.to_csv(out_csv, index=False)
    print(f"\n=== Classical vs. Quantum Comparison (also saved to {out_csv}) ===")
    print(comparison.to_string(index=False))


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full DDI prediction pipeline.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
