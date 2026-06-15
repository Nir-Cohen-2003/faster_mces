"""
Benchmark: accurate MCES (default solver) vs. batched cuOpt solver.

Compares the existing accurate MCES computation (solver="default") against
batching independent pair-wise MCES ILPs into one large block-diagonal problem
and solving it with NVIDIA cuOpt.

Reads the first N molecules from ``tests/dsstox_smiles_medium.csv``,
computes all pairwise symmetric distances with both methods, validates that
the returned matrices agree, and reports wall-clock timings.

For the N=1000 case CPU and GPU utilization are sampled during both runs and
summarized; explanatory messages are emitted when utilization is low.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pulp

# Make the source package importable when running from benchmarks/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mces_splitting.bounds import mces_lower_bound_symmetric  # noqa: E402
from mces_splitting.mces import (  # noqa: E402
    _cached_construct_graph,
    calculate_mces_distances,
)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import psutil

    HAS_PSUTIL = True
except Exception:  # pragma: no cover
    HAS_PSUTIL = False

try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:  # pragma: no cover
    HAS_PYNVML = False

try:
    from cuopt.pulp import CuOptSolver

    HAS_CUOPT_SOLVER = True
except Exception:  # pragma: no cover
    HAS_CUOPT_SOLVER = False

# pulp.CUOPT may also be exposed directly depending on the cuopt/PuLP version.
HAS_PULP_CUOPT = hasattr(pulp, "CUOPT")

DATA_PATH = os.path.join(REPO_ROOT, "tests", "dsstox_smiles_medium.csv")
SIZES = [10, 100, 1000]
DEFAULT_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------
class ResourceMonitor:
    """Sample CPU and (optionally) GPU utilization in a background thread."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0) -> None:
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cpu_samples: List[float] = []
        self.gpu_samples: List[float] = []
        self._gpu_handle: Optional[int] = None
        if HAS_PYNVML:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu_handle = None

    def _sample(self) -> Tuple[float, Optional[float]]:
        cpu = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
        gpu: Optional[float] = None
        if self._gpu_handle is not None:
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu = float(rates.gpu)
            except Exception:
                gpu = None
        return cpu, gpu

    def _run(self) -> None:
        # Prime CPU percent measurement so the first sample is meaningful.
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            cpu, gpu = self._sample()
            self.cpu_samples.append(cpu)
            if gpu is not None:
                self.gpu_samples.append(gpu)
            time.sleep(self.interval)

    def start(self) -> None:
        self.cpu_samples.clear()
        self.gpu_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 0.5)

    def summary(self) -> Dict[str, Optional[float]]:
        def stats(samples: List[float]) -> Dict[str, Optional[float]]:
            if not samples:
                return {"avg": None, "max": None, "min": None}
            arr = np.asarray(samples)
            return {"avg": float(np.mean(arr)), "max": float(np.max(arr)), "min": float(np.min(arr))}

        cpu_stats = stats(self.cpu_samples)
        gpu_stats = stats(self.gpu_samples)
        return {
            "cpu_avg": cpu_stats["avg"],
            "cpu_max": cpu_stats["max"],
            "cpu_min": cpu_stats["min"],
            "gpu_avg": gpu_stats["avg"],
            "gpu_max": gpu_stats["max"],
            "gpu_min": gpu_stats["min"],
        }


def explain_utilization(summary: Dict[str, Optional[float]], method: str) -> List[str]:
    """Return human-readable explanations for low CPU/GPU utilization."""
    messages: List[str] = []
    cpu_avg = summary.get("cpu_avg")
    gpu_avg = summary.get("gpu_avg")

    if cpu_avg is None:
        messages.append("  - CPU utilization could not be tracked (psutil not available).")
    elif cpu_avg < 50.0:
        messages.append(
            f"  - CPU utilization was low (avg {cpu_avg:.1f}%). "
            f"For '{method}', this can mean the work is bound by solver/GPU data "
            f"transfer, Python overhead, or that the problem is too small to keep "
            f"all cores busy."
        )
    else:
        messages.append(f"  - CPU utilization was healthy (avg {cpu_avg:.1f}%).")

    if HAS_PYNVML:
        if gpu_avg is None:
            messages.append("  - GPU utilization could not be sampled (no handle).")
        elif gpu_avg < 50.0:
            messages.append(
                f"  - GPU utilization was low (avg {gpu_avg:.1f}%). "
                f"For cuOpt this usually indicates that the batched model is too "
                f"small, that host-to-device transfer or Python model-building "
                f"overhead dominates, or that the solver is spending most of its "
                f"time on CPU pre-processing rather than GPU kernels."
            )
        else:
            messages.append(f"  - GPU utilization was healthy (avg {gpu_avg:.1f}%).")
    else:
        messages.append("  - GPU utilization could not be tracked (pynvml not available).")

    return messages


# ---------------------------------------------------------------------------
# Batched cuOpt MCES
# ---------------------------------------------------------------------------
def _atoms_match(g1, g2, e1: Tuple[int, int], e2: Tuple[int, int]) -> bool:
    """Return True if the atom labels at the endpoints of e1 and e2 agree."""
    return (
        g1.nodes[e1[0]]["atom"] == g2.nodes[e2[0]]["atom"]
        and g1.nodes[e1[1]]["atom"] == g2.nodes[e2[1]]["atom"]
    ) or (
        g1.nodes[e1[1]]["atom"] == g2.nodes[e2[0]]["atom"]
        and g1.nodes[e1[0]]["atom"] == g2.nodes[e2[1]]["atom"]
    )


def build_batched_mces_problem(
    graphs: List,
    pairs: List[Tuple[int, int]],
) -> Tuple[pulp.LpProblem, Dict[Tuple[int, int], List[Tuple[float, pulp.LpVariable]]]]:
    """Build one combined LP containing independent MCES sub-problems.

    Variables and constraints for each pair are namespaced so the resulting
    matrix is block-diagonal.  The objective is the sum of the individual pair
    objectives; because the blocks are independent, optimizing the global sum
    optimizes each pair individually.

    Returns
    -------
    problem : pulp.LpProblem
        The combined minimization problem.
    pair_objective_terms : dict
        Mapping ``(i, j) -> [(weight, c_variable), ...]`` so the solved
        distance for a pair can be recovered as ``sum(w * c.varValue)``.
    """
    problem = pulp.LpProblem("BatchMCES", pulp.LpMinimize)
    pair_objective_terms: Dict[Tuple[int, int], List[Tuple[float, pulp.LpVariable]]] = {}

    for pair_idx, (i, j) in enumerate(pairs):
        pid = f"p{pair_idx}_{i}_{j}"
        g1 = graphs[i]
        g2 = graphs[j]

        # Node-pair variables y[(n1, n2)]
        y: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for n1 in g1.nodes:
            for n2 in g2.nodes:
                if g1.nodes[n1]["atom"] == g2.nodes[n2]["atom"]:
                    name = f"y_{pid}_{n1}_{n2}"
                    y[(n1, n2)] = pulp.LpVariable(
                        name, lowBound=0, upBound=1, cat=pulp.LpInteger
                    )

        # Edge-pair variables c[(e1, e2)] and not-mapped edge variables.
        c: Dict[Tuple, pulp.LpVariable] = {}
        edge_weights: Dict[Tuple, float] = {}

        for e1 in g1.edges:
            for e2 in g2.edges:
                if _atoms_match(g1, g2, e1, e2):
                    name = f"c_{pid}_{e1[0]}_{e1[1]}_{e2[0]}_{e2[1]}"
                    c[(e1, e2)] = pulp.LpVariable(
                        name, lowBound=0, upBound=1, cat=pulp.LpInteger
                    )
                    edge_weights[(e1, e2)] = (
                        max(g1[e1[0]][e1[1]]["weight"], g2[e2[0]][e2[1]]["weight"])
                        - min(g1[e1[0]][e1[1]]["weight"], g2[e2[0]][e2[1]]["weight"])
                    )

        for e1 in g1.edges:
            name = f"c_{pid}_{e1[0]}_{e1[1]}_nm"
            c[(e1, -1)] = pulp.LpVariable(
                name, lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            edge_weights[(e1, -1)] = g1[e1[0]][e1[1]]["weight"]

        for e2 in g2.edges:
            name = f"c_{pid}_nm_{e2[0]}_{e2[1]}"
            c[(-1, e2)] = pulp.LpVariable(
                name, lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            edge_weights[(-1, e2)] = g2[e2[0]][e2[1]]["weight"]

        # Constraints copied from MCES_ILP, namespaced to this pair.
        # 1. Each node of G1 maps to at most one node of G2.
        for n1 in g1.nodes:
            h = [y[(n1, n2)] for n2 in g2.nodes if (n1, n2) in y]
            if h:
                problem += (
                    pulp.lpSum(h) <= 1,
                    f"G1Node_{pid}_{n1}",
                )

        # 2. Each node of G2 maps to at most one node of G1.
        for n2 in g2.nodes:
            h = [y[(n1, n2)] for n1 in g1.nodes if (n1, n2) in y]
            if h:
                problem += (
                    pulp.lpSum(h) <= 1,
                    f"G2Node_{pid}_{n2}",
                )

        # 3. Every edge in G1 is mapped or marked not-mapped.
        for e1 in g1.edges:
            ls = [c[(e1, e2)] for e2 in g2.edges if (e1, e2) in c]
            problem += (
                pulp.lpSum(ls) + c[(e1, -1)] == 1,
                f"G1Edge_{pid}_{e1[0]}_{e1[1]}",
            )

        # 4. Every edge in G2 is mapped or marked not-mapped.
        for e2 in g2.edges:
            ls = [c[(e1, e2)] for e1 in g1.edges if (e1, e2) in c]
            problem += (
                pulp.lpSum(ls) + c[(-1, e2)] == 1,
                f"G2Edge_{pid}_{e2[0]}_{e2[1]}",
            )

        # 5. Edge mappings must be consistent with node mappings (G1 side).
        for n1 in g1.nodes:
            for e2 in g2.edges:
                ls = []
                for k in g1.neighbors(n1):
                    key1 = (tuple([n1, k]), e2)
                    key2 = (tuple([k, n1]), e2)
                    if key1 in c:
                        ls.append(c[key1])
                    elif key2 in c:
                        ls.append(c[key2])
                rs = []
                if g1.nodes[n1]["atom"] == g2.nodes[e2[0]]["atom"]:
                    rs.append(y[(n1, e2[0])])
                if g1.nodes[n1]["atom"] == g2.nodes[e2[1]]["atom"]:
                    rs.append(y[(n1, e2[1])])
                if ls:
                    problem += (
                        pulp.lpSum(ls) <= pulp.lpSum(rs),
                        f"EdgeNodeConsG1_{pid}_{n1}_{e2[0]}_{e2[1]}",
                    )

        # 6. Edge mappings must be consistent with node mappings (G2 side).
        for n2 in g2.nodes:
            for e1 in g1.edges:
                ls = []
                for k in g2.neighbors(n2):
                    key1 = (e1, tuple([n2, k]))
                    key2 = (e1, tuple([k, n2]))
                    if key1 in c:
                        ls.append(c[key1])
                    elif key2 in c:
                        ls.append(c[key2])
                rs = []
                if g2.nodes[n2]["atom"] == g1.nodes[e1[0]]["atom"]:
                    rs.append(y[(e1[0], n2)])
                if g2.nodes[n2]["atom"] == g1.nodes[e1[1]]["atom"]:
                    rs.append(y[(e1[1], n2)])
                if ls:
                    problem += (
                        pulp.lpSum(ls) <= pulp.lpSum(rs),
                        f"EdgeNodeConsG2_{pid}_{e1[0]}_{e1[1]}_{n2}",
                    )

        pair_objective_terms[(i, j)] = [
            (edge_weights[ep], c[ep]) for ep in edge_weights
        ]

    # Global objective = sum of independent pair objectives.
    all_terms = [
        w * var for terms in pair_objective_terms.values() for (w, var) in terms
    ]
    problem += pulp.lpSum(all_terms), "Global_Objective"

    return problem, pair_objective_terms


def extract_batched_results(
    pair_objective_terms: Dict[Tuple[int, int], List[Tuple[float, pulp.LpVariable]]],
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Recover the solved objective value for each pair and cap at threshold."""
    results = []
    for (i, j), terms in pair_objective_terms.items():
        val = 0.0
        for w, var in terms:
            if var.varValue is None:
                val = threshold
                break
            val += w * var.varValue
        val = min(val, threshold)
        results.append((i, j, val))
    return results


def solve_mces_batch_cuopt(
    graphs: List,
    pairs: List[Tuple[int, int]],
    threshold: float,
    time_limit: int = 600,
    mip_gap: float = 1e-4,
) -> List[Tuple[int, int, float]]:
    """Solve a batch of independent MCES problems with cuOpt."""
    problem, pair_objective_terms = build_batched_mces_problem(graphs, pairs)

    if HAS_CUOPT_SOLVER:
        solver = CuOptSolver(time_limit=time_limit, mip_absolute_gap=mip_gap)
    elif HAS_PULP_CUOPT:
        solver = pulp.CUOPT(msg=0)
    else:
        raise RuntimeError("cuOpt solver is not available.")

    problem.solve(solver)

    if problem.status != pulp.LpStatusOptimal:
        warnings.warn(
            f"cuOpt batch solve returned non-optimal status: {pulp.LpStatus[problem.status]}"
        )

    return extract_batched_results(pair_objective_terms, threshold)


def calculate_mces_distances_cuopt(
    smiles_list: List[str],
    threshold: int = DEFAULT_THRESHOLD,
    cuopt_batch_size: int = 500,
    n_jobs: int = -1,
    time_limit: int = 600,
) -> np.ndarray:
    """Compute symmetric MCES distances using bound filtering + batched cuOpt.

    The returned matrix matches the semantics of ``calculate_mces_distances``:
    lower-bound estimates are returned for pairs whose bound is >= threshold,
    otherwise the exact distance (capped at ``threshold``) is returned.
    """
    smiles_list = list(smiles_list)
    if n_jobs == -1:
        import multiprocessing

        n_jobs = multiprocessing.cpu_count()

    # 1. Fast lower-bound filtering.
    bounds = mces_lower_bound_symmetric(smiles_list)
    distance_matrix = bounds.copy()

    selected = np.argwhere(distance_matrix < threshold)
    selected = selected[selected[:, 0] < selected[:, 1]]
    pairs: List[Tuple[int, int]] = [(int(i), int(j)) for i, j in selected]

    if not pairs:
        return distance_matrix

    # 2. Build all graphs once.
    graphs = [_cached_construct_graph(s) for s in smiles_list]

    # 3. Solve exact distances in cuOpt batches.
    all_results: List[Tuple[int, int, float]] = []
    num_batches = (len(pairs) + cuopt_batch_size - 1) // cuopt_batch_size
    for b_idx, batch_start in enumerate(range(0, len(pairs), cuopt_batch_size)):
        batch_pairs = pairs[batch_start : batch_start + cuopt_batch_size]
        print(
            f"    cuOpt batch {b_idx + 1}/{num_batches}: {len(batch_pairs)} pairs",
            flush=True,
        )
        results = solve_mces_batch_cuopt(
            graphs, batch_pairs, threshold=threshold, time_limit=time_limit
        )
        all_results.extend(results)

    # 4. Fill symmetric matrix.
    for i, j, dist in all_results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist

    return distance_matrix


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def load_smiles(n: int) -> List[str]:
    """Load the first ``n`` SMILES from the dsstox file."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dsstox file at {DATA_PATH}")

    df = pl.read_csv(DATA_PATH)
    col = "MS_READY_SMILES"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in {DATA_PATH}, got {df.columns}")

    smiles = df[col].head(n).to_list()
    smiles = [s for s in smiles if s and isinstance(s, str)]
    if len(smiles) < n:
        warnings.warn(
            f"Requested {n} molecules but only {len(smiles)} valid SMILES found."
        )
    return smiles


def validate_results(
    default_mat: np.ndarray,
    cuopt_mat: np.ndarray,
    size: int,
    tol: float = 1e-3,
) -> Tuple[bool, Optional[str]]:
    """Check that the two distance matrices are numerically equivalent."""
    if default_mat.shape != cuopt_mat.shape:
        return False, f"shape mismatch: {default_mat.shape} vs {cuopt_mat.shape}"

    diff = np.abs(default_mat - cuopt_mat)
    max_diff = float(np.nanmax(diff))
    if max_diff > tol:
        n_bad = int(np.sum(diff > tol))
        return False, f"{n_bad}/{size * size} entries differ by > {tol} (max {max_diff:.4f})"
    return True, f"max difference {max_diff:.4f}"


def run_benchmark(
    size: int,
    run_default: bool = True,
    run_cuopt: bool = True,
    monitor_resources: bool = False,
    cuopt_batch_size: int = 500,
) -> Dict:
    """Run a single benchmark size and return timing/validation results."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark size: {size} molecules ({size * (size - 1) // 2} pairs)")
    print(f"{'=' * 60}")

    smiles = load_smiles(size)
    size = len(smiles)
    print(f"Loaded {size} valid molecules.")

    result: Dict = {"size": size, "pairs": size * (size - 1) // 2}

    # --- Default solver -----------------------------------------------------
    if run_default:
        print("\n[default solver] Starting accurate MCES (solver='default') ...")
        monitor = ResourceMonitor() if monitor_resources else None
        if monitor:
            monitor.start()

        t0 = time.perf_counter()
        default_mat = calculate_mces_distances(
            smiles, smiles_list2=None, solver="default"
        )
        default_time = time.perf_counter() - t0

        if monitor:
            monitor.stop()
        result["default_time"] = default_time
        result["default_ok"] = True
        print(f"[default solver] Done in {default_time:.2f}s")
        if monitor:
            result["default_resources"] = monitor.summary()
            for line in explain_utilization(result["default_resources"], "default"):
                print(line)
    else:
        default_mat = None
        result["default_time"] = None
        result["default_ok"] = False

    # --- cuOpt solver -------------------------------------------------------
    if run_cuopt and (HAS_CUOPT_SOLVER or HAS_PULP_CUOPT):
        print("\n[cuOpt solver] Starting batched cuOpt MCES ...")
        monitor = ResourceMonitor() if monitor_resources else None
        if monitor:
            monitor.start()

        t0 = time.perf_counter()
        try:
            cuopt_mat = calculate_mces_distances_cuopt(
                smiles,
                threshold=DEFAULT_THRESHOLD,
                cuopt_batch_size=cuopt_batch_size,
            )
            cuopt_time = time.perf_counter() - t0
            cuopt_ok = True
            error_msg = None
        except Exception as exc:
            cuopt_time = time.perf_counter() - t0
            cuopt_mat = None
            cuopt_ok = False
            error_msg = str(exc)

        if monitor:
            monitor.stop()

        result["cuopt_time"] = cuopt_time
        result["cuopt_ok"] = cuopt_ok
        result["cuopt_error"] = error_msg
        print(f"[cuOpt solver] Done in {cuopt_time:.2f}s")
        if monitor:
            result["cuopt_resources"] = monitor.summary()
            for line in explain_utilization(result["cuopt_resources"], "cuOpt"):
                print(line)
        if error_msg:
            print(f"[cuOpt solver] ERROR: {error_msg}")
    else:
        cuopt_mat = None
        result["cuopt_time"] = None
        result["cuopt_ok"] = False
        result["cuopt_error"] = "cuOpt not available"
        if not run_cuopt:
            print("\n[cuOpt solver] skipped by user")
        else:
            print("\n[cuOpt solver] skipped (cuOpt not installed)")

    # --- Validation ---------------------------------------------------------
    if default_mat is not None and cuopt_mat is not None:
        valid, msg = validate_results(default_mat, cuopt_mat, size)
        result["validation_passed"] = valid
        result["validation_message"] = msg
        status = "PASS" if valid else "FAIL"
        print(f"\n[validation] {status}: {msg}")
    else:
        result["validation_passed"] = None
        result["validation_message"] = "not run (one or both methods failed/skipped)"
        print("\n[validation] skipped")

    return result


def print_summary(results: List[Dict]) -> None:
    """Print a formatted table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Size':>6} | {'Pairs':>10} | {'Default (s)':>12} | {'cuOpt (s)':>12} | {'Speedup':>10} | {'Validation':>12}"
    )
    print("-" * 80)
    for r in results:
        size = r["size"]
        pairs = r["pairs"]
        default_t = r.get("default_time")
        cuopt_t = r.get("cuopt_time")
        speedup = (
            default_t / cuopt_t if (default_t and cuopt_t and cuopt_t > 0) else None
        )
        valid = r.get("validation_passed")
        valid_str = {True: "PASS", False: "FAIL", None: "n/a"}.get(valid, "n/a")
        print(
            f"{size:>6} | {pairs:>10} | "
            f"{default_t:>12.2f} | {cuopt_t:>12.2f} | "
            f"{speedup:>10.2f}x | {valid_str:>12}"
        )
    print("=" * 80)


def save_csv(results: List[Dict], path: str) -> None:
    """Save a concise CSV summary to ``path``."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "size",
                "pairs",
                "default_time_s",
                "cuopt_time_s",
                "speedup",
                "validation_passed",
                "validation_message",
                "cuopt_error",
            ]
        )
        for r in results:
            default_t = r.get("default_time")
            cuopt_t = r.get("cuopt_time")
            speedup = (
                default_t / cuopt_t if (default_t and cuopt_t and cuopt_t > 0) else None
            )
            writer.writerow(
                [
                    r["size"],
                    r["pairs"],
                    default_t,
                    cuopt_t,
                    speedup,
                    r.get("validation_passed"),
                    r.get("validation_message"),
                    r.get("cuopt_error"),
                ]
            )
    print(f"Saved summary to {path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark default vs. batched cuOpt MCES distances."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SIZES,
        help=f"Molecule counts to benchmark (default: {SIZES})",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="Skip the default-solver baseline.",
    )
    parser.add_argument(
        "--skip-cuopt",
        action="store_true",
        help="Skip the cuOpt solver.",
    )
    parser.add_argument(
        "--monitor-at-size",
        type=int,
        default=1000,
        help="Enable CPU/GPU monitoring when size equals this value (default: 1000).",
    )
    parser.add_argument(
        "--cuopt-batch-size",
        type=int,
        default=500,
        help="Number of pair-wise MCES problems to combine in one cuOpt batch (default: 500).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(REPO_ROOT, "benchmarks", "benchmark_cuopt_results.csv"),
        help="Path to write CSV summary.",
    )
    args = parser.parse_args()

    print("cuOpt availability:")
    print(f"  - cuopt.pulp.CuOptSolver: {'yes' if HAS_CUOPT_SOLVER else 'no'}")
    print(f"  - pulp.CUOPT: {'yes' if HAS_PULP_CUOPT else 'no'}")
    print(f"  - psutil (CPU monitoring): {'yes' if HAS_PSUTIL else 'no'}")
    print(f"  - pynvml (GPU monitoring): {'yes' if HAS_PYNVML else 'no'}")

    results: List[Dict] = []
    for size in args.sizes:
        monitor = size == args.monitor_at_size
        r = run_benchmark(
            size=size,
            run_default=not args.skip_default,
            run_cuopt=not args.skip_cuopt,
            monitor_resources=monitor,
            cuopt_batch_size=args.cuopt_batch_size,
        )
        results.append(r)

    print_summary(results)
    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
