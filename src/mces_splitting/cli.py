"""Command-line interface for MCES-based dataset splitting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .dataset_splitting import split_dataset


def _read_smiles(path: str | Path) -> list[str]:
    """Read SMILES from a .smi or .txt file (one SMILES per line)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    smiles: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line)

    if not smiles:
        raise ValueError(f"No valid SMILES found in {path}")

    return smiles


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write a list of strings to a file, one per line."""
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``mces-split``."""
    parser = argparse.ArgumentParser(
        description=(
            "Split a molecular dataset into train/validation/test sets using "
            "MCES lower-bound distances. Supports threshold-based (connected "
            "components) or UMAP-based splitting."
        )
    )
    parser.add_argument(
        "input",
        help="Input file containing one SMILES string per line (.smi or .txt).",
    )
    parser.add_argument(
        "--method",
        choices=["threshold", "umap"],
        default="threshold",
        help="Splitting strategy (default: threshold).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Target fraction for the validation set (default: 0.1).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Target fraction for the test set (default: 0.1).",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.7,
        help="Minimum required size ratio for validation/test vs target (default: 0.7).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where output split files will be written (default: current directory).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files (default: input file stem).",
    )
    parser.add_argument(
        "--mces-matrix-save-path",
        default=None,
        help="Optional path to save the lower-bound distance matrix as a .npy file.",
    )

    # Threshold-specific options
    threshold_group = parser.add_argument_group("Threshold-based splitting options")
    threshold_group.add_argument(
        "--initial-distinction-threshold",
        type=int,
        default=10,
        help="Starting MCES distinction threshold (default: 10).",
    )
    threshold_group.add_argument(
        "--min-distinction-threshold",
        type=int,
        default=2,
        help="Lowest MCES distinction threshold to try (default: 2).",
    )
    threshold_group.add_argument(
        "--threshold-step",
        type=int,
        default=-1,
        help="Step size when lowering the threshold (default: -1).",
    )

    # UMAP-specific options
    umap_group = parser.add_argument_group("UMAP-based splitting options")
    umap_group.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="UMAP embedding dimensionality (default: 2).",
    )
    umap_group.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="UMAP n_neighbors (default: capped at n - 1).",
    )
    umap_group.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1).",
    )
    umap_group.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=None,
        help="HDBSCAN min_cluster_size (default: adaptive based on dataset size).",
    )
    umap_group.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=1,
        help="HDBSCAN min_samples (default: 1).",
    )

    args = parser.parse_args(argv)

    smiles = _read_smiles(args.input)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix if args.output_prefix else input_path.stem

    kwargs: dict[str, object] = {}
    if args.method == "threshold":
        kwargs.update(
            {
                "initial_distinction_threshold": args.initial_distinction_threshold,
                "min_distinction_threshold": args.min_distinction_threshold,
                "threshold_step": args.threshold_step,
            }
        )
    else:
        umap_kwargs: dict[str, object] = {
            "n_components": args.n_components,
            "min_dist": args.min_dist,
        }
        if args.n_neighbors is not None:
            umap_kwargs["n_neighbors"] = args.n_neighbors

        hdbscan_kwargs: dict[str, object] = {
            "min_samples": args.hdbscan_min_samples,
        }
        if args.hdbscan_min_cluster_size is not None:
            hdbscan_kwargs["min_cluster_size"] = args.hdbscan_min_cluster_size

        kwargs["hdbscan_kwargs"] = hdbscan_kwargs
        kwargs.update(umap_kwargs)

    result = split_dataset(
        smiles,
        method=args.method,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        min_ratio=args.min_ratio,
        random_state=args.random_state,
        mces_matrix_save_path=args.mces_matrix_save_path,
        **kwargs,
    )

    train_path = output_dir / f"{prefix}_train.smi"
    val_path = output_dir / f"{prefix}_val.smi"
    test_path = output_dir / f"{prefix}_test.smi"

    _write_lines(train_path, result["train"])
    _write_lines(val_path, result["validation"])
    _write_lines(test_path, result["test"])

    print(f"Wrote {len(result['train'])} train SMILES to    {train_path}")
    print(f"Wrote {len(result['validation'])} validation SMILES to {val_path}")
    print(f"Wrote {len(result['test'])} test SMILES to     {test_path}")
    if "threshold" in result:
        print(f"Final distinction threshold: {result['threshold']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
