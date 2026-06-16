"""
Benchmark MCES lower bound, upper bound, and exact MCES across increasing dataset sizes.

Usage examples:
    pixi run python benchmarks/benchmark_bounds.py
    pixi run python benchmarks/benchmark_bounds.py --max-n 1000
    pixi run python benchmarks/benchmark_bounds.py --sizes 50 100 200 --n-jobs 8

Outputs:
    - Printed summary table to stdout
    - JSON results file (default: benchmarks/benchmark_bounds_results.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence

import numpy as np
import polars as pl
import psutil
from rdkit import Chem

# Package APIs
from mces_splitting.bounds import mces_lower_bound_symmetric
from mces_splitting.bounds import mces_upper_bound_symmetric
from mces_splitting.mces import exact_mces_for_list_of_pairs


# ---------------------------------------------------------------------------
# CPU monitoring helper
# ---------------------------------------------------------------------------
class CPUMonitor:
    """Sample per-core CPU usage in a background thread while work runs."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: list[list[float]] = []
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        # psutil.cpu_percent needs a baseline call; the first sample is often 0.0.
        while self._running.is_set():
            sample = psutil.cpu_percent(percpu=True, interval=None)
            self._samples.append(sample)
            time.sleep(self.interval)

    def start(self) -> None:
        self._samples.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if not self._samples:
            return {
                "total_cpu_percent": None,
                "mean_per_core_percent": [],
                "max_single_core_percent": None,
                "n_samples": 0,
            }
        arr = np.array(self._samples)
        mean_per_core = arr.mean(axis=0).tolist()
        return {
            "total_cpu_percent": float(sum(mean_per_core)),
            "mean_per_core_percent": mean_per_core,
            "max_single_core_percent": float(max(mean_per_core)),
            "n_samples": len(self._samples),
        }


def timed_with_cpu(
    func: Callable[[], Any], interval: float = 0.1
) -> tuple[Any, float, dict[str, Any]]:
    """Run ``func`` and return (result, wall_seconds, cpu_stats)."""
    monitor = CPUMonitor(interval=interval)
    # Warm-up psutil baseline.
    psutil.cpu_percent(percpu=True, interval=None)
    time.sleep(0.05)
    monitor.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        cpu_stats = monitor.stop()
    return result, elapsed, cpu_stats


# ---------------------------------------------------------------------------
# Upper-bound benchmark (in-process, OpenMP parallelised in C++)
# ---------------------------------------------------------------------------
def upper_bound_matrix(
    smiles: list[str],
    n_jobs: int = -1,
    num_starts: int = 100,
    connected: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute full symmetric upper-bound matrix in-process via the C++ OpenMP implementation."""
    n = len(smiles)
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    n_jobs = max(1, min(n_jobs, cpu_count()))

    if verbose:
        print(
            f"  upper bound: {n * (n - 1) // 2} pairs using up to {n_jobs} OpenMP threads"
        )

    def run() -> np.ndarray:
        # Configure OpenMP thread count for the in-process C++ computation.
        # Only set OMP_NUM_THREADS if the caller explicitly asked for a thread
        # count and the environment variable is not already set.
        if n_jobs > 0 and "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(n_jobs)
        return mces_upper_bound_symmetric(
            smiles, connected=connected, num_starts=num_starts
        )

    mat, elapsed, cpu = timed_with_cpu(run)
    return mat, elapsed, cpu


# ---------------------------------------------------------------------------
# Lower-bound benchmark
# ---------------------------------------------------------------------------
def lower_bound_matrix(smiles: list[str]) -> tuple[np.ndarray, float, dict[str, Any]]:
    def run() -> np.ndarray:
        return mces_lower_bound_symmetric(smiles)

    mat, elapsed, cpu = timed_with_cpu(run)
    return mat, elapsed, cpu


# ---------------------------------------------------------------------------
# Exact MCES benchmark (production pipeline, threshold based)
# ---------------------------------------------------------------------------
@contextmanager
def _suppress_fd_output() -> Generator[None, None, None]:
    """Redirect OS file descriptors 1 and 2 to /dev/null to silence C-level stdout."""
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(original_stdout)
        os.close(original_stderr)
        os.close(devnull)


def exact_matrix(
    smiles: list[str],
    n_jobs: int = -1,
    threshold: int = 10,
    solver: str = "default",
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Compute exact MCES matrix using a ProcessPoolExecutor parallel function.

    First filters pairs using the lower-bound matrix, then calls
    ``exact_mces_for_list_of_pairs`` in parallel to compute exact distances
    only for pairs that may be below the threshold.
    """

    def run() -> np.ndarray:
        # 1. Lower-bound filter (cheap, already parallelised internally).
        lb_mat = mces_lower_bound_symmetric(smiles)
        n = len(smiles)
        mat = lb_mat.copy()

        # 2. Determine which pairs need exact evaluation.
        pairs_needing_exact = [
            (int(i), int(j)) for i, j in zip(*np.where(mat < threshold)) if i < j
        ]

        if not pairs_needing_exact:
            return mat

        # 3. Exact MCES in parallel via ProcessPoolExecutor (library function).
        with _suppress_fd_output():
            exact_results = exact_mces_for_list_of_pairs(
                smiles,
                smiles,
                pairs_needing_exact,
                n_jobs=n_jobs,
                batch_size=20,
                threshold=threshold,
                solver=solver,
            )

        for i, j, distance in exact_results:
            if distance is not None:
                mat[i, j] = distance
                mat[j, i] = distance

        return mat

    mat, elapsed, cpu = timed_with_cpu(run)
    return mat, elapsed, cpu


# ---------------------------------------------------------------------------
# Accuracy comparison
# ---------------------------------------------------------------------------
def compare_to_exact(
    exact_mat: np.ndarray,
    lower_mat: np.ndarray,
    upper_mat: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    # Only compare pairs where the exact pipeline returned a value strictly below the cap.
    triu = np.triu_indices(len(exact_mat), k=1)
    exact_vals = exact_mat[triu]
    lower_vals = lower_mat[triu]
    upper_vals = upper_mat[triu]
    mask = exact_vals < threshold - 1e-6

    if not np.any(mask):
        return {
            "n_comparable_pairs": 0,
            "lower_mae": None,
            "upper_mae": None,
            "lower_max_error": None,
            "upper_max_error": None,
            "lower_exact_pct": None,
            "upper_exact_pct": None,
        }

    lower_diff = np.abs(lower_vals[mask] - exact_vals[mask])
    upper_diff = np.abs(upper_vals[mask] - exact_vals[mask])

    return {
        "n_comparable_pairs": int(np.count_nonzero(mask)),
        "lower_mae": float(np.mean(lower_diff)),
        "upper_mae": float(np.mean(upper_diff)),
        "lower_max_error": float(np.max(lower_diff)),
        "upper_max_error": float(np.max(upper_diff)),
        "lower_exact_pct": float(np.mean(lower_diff < 1e-6) * 100),
        "upper_exact_pct": float(np.mean(upper_diff < 1e-6) * 100),
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------
def load_smiles(path: str | Path, sanitize: bool = True) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pl.scan_csv(path).select("MS_READY_SMILES").collect()
    smiles = df.to_series().to_list()
    smiles = [s for s in smiles if isinstance(s, str) and s.strip()]
    if not smiles:
        raise ValueError(f"No valid SMILES found in {path}")

    if sanitize:
        sanitized: list[str] = []
        skipped = 0
        for s in smiles:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    sanitized.append(Chem.MolToSmiles(mol))
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if skipped:
            print(
                f"Sanitization: skipped {skipped} malformed SMILES out of {len(smiles)}"
            )
        smiles = sanitized

    if not smiles:
        raise ValueError(f"No valid SMILES found in {path} after sanitization")
    return smiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MCES lower/upper bounds vs exact MCES"
    )
    parser.add_argument(
        "--dataset",
        default="tests/dsstox_smiles_medium.csv",
        help="Path to CSV with an MS_READY_SMILES column (default: tests/dsstox_smiles_medium.csv)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        help="Space-separated size ladder (default: 50 100 200 500 1000 2000 5000 10000)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=10000,
        help="Maximum dataset size to run (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/benchmark_bounds_results.json",
        help="Path for JSON results (default: benchmarks/benchmark_bounds_results.json)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = all cores, default: -1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10000,
        help="Threshold passed to calculate_mces_distances (default: 10000, to get the exact values)",
    )
    parser.add_argument(
        "--num-starts",
        type=int,
        default=100,
        help="num_starts for the clique-based upper bound (default: 100)",
    )
    parser.add_argument(
        "--exact-up-to",
        type=int,
        default=200,
        help="Run exact MCES comparison only for n <= this value (default: 200)",
    )
    parser.add_argument(
        "--solver",
        default="default",
        help="ILP solver for exact MCES (default: default/CBC)",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable RDKit sanitization of input SMILES (default: sanitize enabled)",
    )
    return parser


def uses_all_cores(
    total_cpu_percent: float | None, n_cores: int, tolerance: float = 0.6
) -> bool | None:
    """Heuristic: total CPU% is at least 60% of 100*n_cores."""
    if total_cpu_percent is None or n_cores <= 0:
        return None
    return total_cpu_percent >= (100.0 * n_cores * tolerance)


def format_time(t: float | None) -> str:
    if t is None:
        return "N/A"
    return f"{t:.3f}s"


def fmt_float(x: float | None, prec: int = 1) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{prec}f}"


def _effective_core_count() -> int:
    """Logical CPUs visible to *this process* (respects affinity/cgroups)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return cpu_count()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path

    smiles = load_smiles(dataset_path, sanitize=not args.no_sanitize)
    n_cores = _effective_core_count()
    n_jobs = n_cores if args.n_jobs == -1 else max(1, args.n_jobs)

    sizes = sorted(set(args.sizes))
    sizes = [n for n in sizes if n <= args.max_n]
    if not sizes:
        print("No sizes to benchmark after applying --max-n.", file=sys.stderr)
        return 1

    print(f"Benchmarking MCES bounds")
    print(f"Dataset: {dataset_path} ({len(smiles)} molecules available)")
    print(f"Sizes: {sizes}")
    print(f"Logical CPUs (cpu_count): {cpu_count()}")
    print(f"Effective CPUs for this process: {n_cores}")
    print(f"Parallel workers: {n_jobs}")
    print(
        f"Exact comparison up to n={args.exact_up_to} using threshold={args.threshold}"
    )
    print("=" * 80)

    results: list[dict[str, Any]] = []

    for n in sizes:
        if n > len(smiles):
            print(
                f"\nSkipping n={n}: not enough molecules in dataset ({len(smiles)} available)"
            )
            continue

        subset = smiles[:n]
        n_pairs = n * (n - 1) // 2
        print(f"\n--- n={n} ({n_pairs} unique pairs) ---")

        entry: dict[str, Any] = {
            "n": n,
            "n_pairs": n_pairs,
            "n_cores": n_cores,
            "n_jobs": n_jobs,
        }

        # Lower bound
        try:
            print("  Running lower bound...")
            lb_mat, lb_time, lb_cpu = lower_bound_matrix(subset)
            entry["lower_bound"] = {
                "time_seconds": lb_time,
                **lb_cpu,
                "uses_all_cores": uses_all_cores(lb_cpu["total_cpu_percent"], n_cores),
            }
            print(
                f"  Lower bound: {format_time(lb_time)} (CPU total={lb_cpu['total_cpu_percent']:.1f}%, max_core={lb_cpu['max_single_core_percent']:.1f}%)"
            )
        except Exception as exc:
            print(f"  Lower bound FAILED: {exc}")
            entry["lower_bound"] = {"error": str(exc)}
            lb_mat = None

        # Upper bound
        try:
            print("  Running upper bound...")
            ub_mat, ub_time, ub_cpu = upper_bound_matrix(
                subset,
                n_jobs=n_jobs,
                num_starts=args.num_starts,
                connected=False,
                verbose=True,
            )
            entry["upper_bound"] = {
                "time_seconds": ub_time,
                **ub_cpu,
                "uses_all_cores": uses_all_cores(ub_cpu["total_cpu_percent"], n_cores),
            }
            print(
                f"  Upper bound: {format_time(ub_time)} (CPU total={ub_cpu['total_cpu_percent']:.1f}%, max_core={ub_cpu['max_single_core_percent']:.1f}%)"
            )
        except Exception as exc:
            print(f"  Upper bound FAILED: {exc}")
            entry["upper_bound"] = {"error": str(exc)}
            ub_mat = None

        # Exact MCES (only up to the configured cutoff)
        if n <= args.exact_up_to:
            try:
                print("  Running exact MCES (production pipeline)...")
                ex_mat, ex_time, ex_cpu = exact_matrix(
                    subset,
                    n_jobs=n_jobs,
                    threshold=args.threshold,
                    solver=args.solver,
                )
                entry["exact"] = {
                    "time_seconds": ex_time,
                    **ex_cpu,
                    "uses_all_cores": uses_all_cores(
                        ex_cpu["total_cpu_percent"], n_cores
                    ),
                }
                print(
                    f"  Exact:       {format_time(ex_time)} (CPU total={ex_cpu['total_cpu_percent']:.1f}%, max_core={ex_cpu['max_single_core_percent']:.1f}%)"
                )

                # Accuracy comparison
                if lb_mat is not None and ub_mat is not None:
                    acc = compare_to_exact(ex_mat, lb_mat, ub_mat, args.threshold)
                    entry["accuracy"] = acc
                    print(
                        f"  Accuracy vs exact (pairs below threshold={args.threshold}): "
                        f"n={acc['n_comparable_pairs']}, "
                        f"lower MAE={fmt_float(acc['lower_mae'], 3)}, upper MAE={fmt_float(acc['upper_mae'], 3)}, "
                        f"lower exact={fmt_float(acc['lower_exact_pct'], 1)}%, upper exact={fmt_float(acc['upper_exact_pct'], 1)}%"
                    )
            except Exception as exc:
                print(f"  Exact MCES FAILED: {exc}")
                entry["exact"] = {"error": str(exc)}
        else:
            print(f"  Skipping exact MCES (n > {args.exact_up_to})")
            entry["exact"] = {"skipped": True, "reason": f"n > {args.exact_up_to}"}

        results.append(entry)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": {
                    "dataset": str(dataset_path),
                    "cpu_count_logical": cpu_count(),
                    "n_cores_effective": n_cores,
                    "n_jobs": n_jobs,
                    "threshold": args.threshold,
                    "num_starts": args.num_starts,
                    "exact_up_to": args.exact_up_to,
                    "solver": args.solver,
                },
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = (
        f"{'n':>6} | {'pairs':>10} | {'lower(s)':>10} | {'upper(s)':>10} | {'exact(s)':>10} | "
        f"{'lb_cpu%':>8} | {'ub_cpu%':>8} | {'ex_cpu%':>8} | "
        f"{'lb_all_cores':>12} | {'ub_all_cores':>12} | {'speedup ex/lb':>13} | {'speedup ex/ub':>13}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        lb = r.get("lower_bound", {})
        ub = r.get("upper_bound", {})
        ex = r.get("exact", {})

        lb_time = lb.get("time_seconds")
        ub_time = ub.get("time_seconds")
        ex_time = ex.get("time_seconds")

        lb_cpu = lb.get("total_cpu_percent")
        ub_cpu = ub.get("total_cpu_percent")
        ex_cpu = ex.get("total_cpu_percent")

        lb_all = lb.get("uses_all_cores")
        ub_all = ub.get("uses_all_cores")

        speedup_lb = ex_time / lb_time if (lb_time and ex_time) else None
        speedup_ub = ex_time / ub_time if (ub_time and ex_time) else None

        print(
            f"{r['n']:>6} | {r['n_pairs']:>10} | "
            f"{format_time(lb_time):>10} | {format_time(ub_time):>10} | {format_time(ex_time):>10} | "
            f"{fmt_float(lb_cpu, 1):>8} | "
            f"{fmt_float(ub_cpu, 1):>8} | "
            f"{fmt_float(ex_cpu, 1):>8} | "
            f"{('yes' if lb_all else 'no') if lb_all is not None else 'N/A':>12} | "
            f"{('yes' if ub_all else 'no') if ub_all is not None else 'N/A':>12} | "
            f"{fmt_float(speedup_lb, 1):>13} | "
            f"{fmt_float(speedup_ub, 1):>13}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
