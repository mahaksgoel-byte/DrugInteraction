"""Run the local DDInter cleaning and structure-preprocessing pipeline.

This pipeline reads files already present under ``data/raw``. It performs no
downloads and never modifies raw inputs. The optional dry run only prints the
stages that would be executed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(seeds: list[int], dry_run: bool = False) -> None:
    commands = [
        ["src/data/preprocess_ddinter.py"],
        ["src/data/build_provenance.py"],
        ["src/data/make_splits.py", "--seeds", *map(str, seeds)],
        ["src/backbone/molecular_features.py"],
        ["src/graph/molecular.py"],
    ]
    for command in commands:
        full = [sys.executable, str(ROOT / command[0]), *command[1:]]
        print("Would run:" if dry_run else "Running:", " ".join(full))
        if not dry_run:
            subprocess.run(full, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--dry-run", action="store_true", help="Print stages without reading/writing dataset artifacts")
    args = parser.parse_args()
    run_pipeline(args.seeds, args.dry_run)


if __name__ == "__main__":
    main()
