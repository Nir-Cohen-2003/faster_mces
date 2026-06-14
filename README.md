# mces_splitting

A Python package for **Maximum Common Edge Subgraph (MCES)** distance calculations and **dataset splitting** for molecular machine learning.

This package provides tools to:

- Compute **exact MCES distances** between molecules using Integer Linear Programming (ILP).
- Compute **fast lower bounds** for MCES distances (delegated to the C++ `fast_mces_lower_bound` extension).
- **Split molecular datasets** into train/validation/test sets while ensuring structural diversity based on MCES distance thresholds.

The dataset splitting strategy works by computing pairwise molecule distances, clustering non-distinct molecules, and distributing clusters across splits to maximize structural separation.

---

## Installation

### From PyPI

```bash
pip install mces_splitting
```

Dependencies include:

- `rdkit`
- `polars`
- `numpy`
- `scipy`
- `networkx`
- `pulp`
- `fast_mces_lower_bound` (C++ extension for lower bound calculations)

### From prebuilt packages

If using `conda`/`mamba`/`micromamba` (replace `conda` with the respective tool):

```bash
conda install -c https://prefix.dev/nir-cohen mces_splitting
```

If using pixi (recommended in general):

```bash
pixi add mces_splitting
```

### Build from source

Install pixi and clone this directory.

To create the package:

```bash
pixi build
```

To install into the default environment of the project:

```bash
pixi install
```

To run tests:

```bash
pixi run splitting_test
```

---

## Quick Start

```python
from mces_splitting import (
    split_dataset_lower_bound_only,
    split_dataset_with_exact_mces,
    split_dataset_umap,
    split_dataset,
    mces_lower_bounds,
)

smiles = ["CCO", "CC=O", "c1ccccc1", "CC(NC)CC1=CC=C(OCO2)C2=C1"]

# Split using fast lower bounds only
train, val, test, threshold = split_dataset_lower_bound_only(
    smiles,
    validation_fraction=0.1,
    test_fraction=0.1,
)

# Or split with exact MCES for critical pairs
train, val, test, threshold = split_dataset_with_exact_mces(
    smiles,
    validation_fraction=0.1,
    test_fraction=0.1,
)

# Or split via UMAP on the MCES lower-bound distance matrix
train, val, test, bounds, embedding = split_dataset_umap(
    smiles,
    validation_fraction=0.1,
    test_fraction=0.1,
    random_state=42,
)

# Or use the high-level dispatcher
result = split_dataset(
    smiles,
    method="umap",  # or "threshold"
    validation_fraction=0.1,
    test_fraction=0.1,
)
```

---

## Package API

### `mces_splitting.mces`

Functions for exact MCES distance computation.

#### `calculate_mces_distances(smiles_list1, smiles_list2=None, n_jobs=-1, batch_size=20, threshold=10, solver="GUROBI")`

Efficiently computes exact MCES distances between all pairs of molecules.

- **Parameters:**
  - `smiles_list1` (`List[str]`): First set of SMILES strings.
  - `smiles_list2` (`Optional[List[str]]`): Second set of SMILES strings. If `None`, computes symmetric distances within `smiles_list1`.
  - `n_jobs` (`int`): Number of parallel jobs. `-1` uses all CPU cores.
  - `batch_size` (`int`): Number of pairs per parallel batch.
  - `threshold` (`int`): Distance threshold. Pairs with bounds above this skip exact calculation.
  - `solver` (`str`): ILP solver to use (`"GUROBI"`, `"default"` for CBC, or `"CUOPT"`).

- **Returns:**
  - `np.ndarray`: Distance matrix where element `[i, j]` is the exact MCES distance.

---

#### `are_close_mols(smiles_list1, smiles_list2=None, *, n_jobs=-1, batch_size=20, solver="GUROBI", symmetric=False)`

Returns a boolean matrix indicating whether each pair of molecules has an MCES distance of `1` or lower.

- **Parameters:**
  - `smiles_list1` (`List[str]`): First set of SMILES strings.
  - `smiles_list2` (`Optional[List[str]]`): Second set of SMILES strings. If `None`, defaults to symmetric comparison.
  - `n_jobs` (`int`): Number of parallel jobs. `-1` uses all CPU cores.
  - `batch_size` (`int`): Number of pairs per parallel batch.
  - `solver` (`str`): Solver backend to use.
  - `symmetric` (`bool`): Ignored; inferred automatically from `smiles_list2`.

- **Returns:**
  - `np.ndarray`: Boolean matrix where `[i, j]` is `True` if distance `<= 1`.

---

#### `exact_mces_for_list_of_pairs(smiles_list1, smiles_list2, pairs, n_jobs=-1, batch_size=20, threshold=10, solver="GUROBI")`

Computes exact MCES distances for a specific list of molecule pairs.

- **Parameters:**
  - `smiles_list1` (`List[str]`): First set of SMILES strings.
  - `smiles_list2` (`List[str]`): Second set of SMILES strings.
  - `pairs` (`List[Tuple[int, int]]`): List of `(i, j)` index pairs to compute.
  - `n_jobs` (`int`): Number of parallel processes. `-1` uses all CPU cores.
  - `batch_size` (`int`): Number of pairs per batch.
  - `threshold` (`int`): Distance threshold for early termination.
  - `solver` (`str`): ILP solver.

- **Returns:**
  - `List[Tuple[int, int, int]]`: List of `(i, j, distance)` tuples for each requested pair. `distance` may be `None` on failure.

---

#### `construct_graph(smiles: str) -> nx.Graph`

Converts a SMILES string into a `networkx.Graph`.

- Nodes have an `"atom"` attribute (element symbol).
- Edges have a `"weight"` attribute (bond type as a double).

---

#### `MCES_ILP(G1, G2, threshold, solver="default", solver_options={}, no_ilp_threshold=False)`

Calculates the exact MCES distance between two molecule graphs using an Integer Linear Program.

- **Parameters:**
  - `G1`, `G2` (`nx.Graph`): Molecule graphs.
  - `threshold` (`float`): Early termination threshold. Use `-1` for no threshold.
  - `solver` (`str`): Solver backend (`"default"`, `"GUROBI"`, `"CUOPT"`).
  - `solver_options` (`dict`): Additional solver options (e.g., `{"threads": 1}`).
  - `no_ilp_threshold` (`bool`): If `True`, always returns exact distance even if above threshold.

- **Returns:**
  - `(float, int)`: `(distance, distance_type)` where `distance_type` is:
    - `1` — exact distance
    - `2` — lower bound (distance exceeded threshold)

---

#### `suppress_output()`

Context manager that suppresses `stdout` and `stderr`. Useful for silencing solver output during batch computations.

---

### `mces_splitting.bounds`

Functions for lower bound estimation and fast distance matrix computation.

#### `mces_lower_bound_symmetric(smiles_list) -> np.ndarray`

Computes a symmetric lower-bound distance matrix for a list of SMILES strings using the fast C++ implementation.

- **Parameters:**
  - `smiles_list` (`Sequence[str]`): List of SMILES strings.

- **Returns:**
  - `np.ndarray`: Symmetric `(n, n)` distance matrix.

---

#### `mces_lower_bound(smiles_list1, smiles_list2) -> np.ndarray`

Computes a lower-bound distance matrix between two lists of SMILES strings using the fast C++ implementation.

- **Parameters:**
  - `smiles_list1` (`Sequence[str]`): First list of SMILES strings.
  - `smiles_list2` (`Sequence[str]`): Second list of SMILES strings.

- **Returns:**
  - `np.ndarray`: `(n1, n2)` distance matrix.

---

#### `filter1(G1, G2) -> float`

Computes a simple degree-based lower bound for the MCES distance between two graphs.

- **Parameters:**
  - `G1`, `G2` (`nx.Graph`): Molecule graphs.

- **Returns:**
  - `float`: Lower bound distance.

---

#### `filter2_from_lib(G1, G2) -> float`

Computes a neighborhood-based lower bound using minimum-weight full matching (Hungarian algorithm). This bound is tighter than `filter1`.

- **Parameters:**
  - `G1`, `G2` (`nx.Graph`): Molecule graphs.

- **Returns:**
  - `float`: Lower bound distance.

---

#### `_get_cost(G1, G2, i, j) -> float`

Internal helper that computes the cost of mapping node `i` in `G1` to node `j` in `G2` based on neighborhood structure.

---

### `mces_splitting.dataset_splitting`

Functions for splitting molecular datasets into train/validation/test sets using MCES-based structural distinctness.

#### `split_dataset(dataset, method="threshold", validation_fraction=0.1, test_fraction=0.1, min_ratio=0.7, random_state=42, mces_matrix_save_path=None, **kwargs)`

High-level dispatcher that selects between the threshold-based (connected components) and UMAP-based splitting strategies.

- **Parameters:**
  - `dataset` (`list[str]`): List of SMILES strings.
  - `method` (`"threshold" | "umap"`): Splitting strategy.
    - `"threshold"`: Build a graph where edges connect non-distinct molecules (distance `< threshold`), find connected components, and distribute clusters across splits while adaptively lowering the threshold if needed.
    - `"umap"`: Embed the MCES lower-bound distance matrix with UMAP, cluster the embedding with HDBSCAN (falling back to k-means if too few clusters are found), and distribute clusters across splits.
  - `validation_fraction` (`float`): Target fraction for validation set.
  - `test_fraction` (`float`): Target fraction for test set.
  - `min_ratio` (`float`): Minimum required size ratio for validation/test vs target.
  - `random_state` (`int`): Random seed passed to the chosen method.
  - `mces_matrix_save_path` (`str | None`): If provided, saves the lower-bound distance matrix to this path as a `.npy` file.
  - `**kwargs`: Forwarded to the underlying splitter. For `"threshold"` this includes `initial_distinction_threshold`, `min_distinction_threshold`, `threshold_step`. For `"umap"` this includes `hdbscan_kwargs` and UMAP arguments such as `n_components`, `n_neighbors`, `min_dist`.

- **Returns:**
  - `dict`: Dictionary with keys `"train"`, `"validation"`, `"test"`. For `"threshold"` it also contains `"threshold"`. For `"umap"` it also contains `"bounds_matrix"` and `"umap_embedding"`.

---

#### `split_dataset_adaptive_threshold(dataset, validation_fraction=0.1, test_fraction=0.1, initial_distinction_threshold=10, min_distinction_threshold=2, threshold_step=-1, min_ratio=0.7, mces_matrix_save_path=None)`

Splits a dataset using **only lower bounds**, adaptively lowering the distinction threshold until a valid split is found.

- **Parameters:**
  - `dataset` (`list[str]`): List of SMILES strings.
  - `validation_fraction` (`float`): Target fraction for validation set.
  - `test_fraction` (`float`): Target fraction for test set.
  - `initial_distinction_threshold` (`int`): Starting MCES threshold for distinctness.
  - `min_distinction_threshold` (`int`): Lowest threshold to try.
  - `threshold_step` (`int`): Step size when lowering threshold (typically `-1`).
  - `min_ratio` (`float`): Minimum required size ratio for validation/test vs target.
  - `mces_matrix_save_path` (`str | None`): If provided, saves the lower-bound matrix to this path as a `.npy` file.

- **Returns:**
  - `Tuple[list[str], list[str], list[str], int]`: `(train_set, validation_set, test_set, final_threshold)`

---

#### `split_dataset_brute_force_exact(dataset, validation_fraction=0.1, test_fraction=0.1, initial_distinction_threshold=10, min_distinction_threshold=2, threshold_step=-1, min_ratio=0.7, max_exact_calculations=None, mces_matrix_save_path=None)`

Splits a dataset using **exact MCES calculations** for all pairs below the threshold. Falls back to lower bounds first, then calculates exact distances for candidate pairs.

- **Parameters:**
  - Same as `split_dataset_adaptive_threshold`, plus:
  - `max_exact_calculations` (`int | None`): Maximum number of exact MCES calculations to perform. If `None`, calculates all candidate pairs.
  - `mces_matrix_save_path` (`str | None`): Saves matrices at each threshold attempt.

- **Returns:**
  - `Tuple[list[str], list[str], list[str], int]`: `(train_set, validation_set, test_set, final_threshold)`

---

#### `split_dataset_with_selective_exact_calculation(dataset, validation_fraction=0.1, test_fraction=0.1, initial_distinction_threshold=10, min_distinction_threshold=2, threshold_step=-1, min_ratio=0.7, max_exact_calculations=1000, mces_matrix_save_path=None)`

Splits a dataset using **strategic exact MCES calculations**. Identifies critical pairs where exact distances are most likely to enable a higher threshold, then selectively computes them within a budget.

- **Parameters:**
  - Same as `split_dataset_adaptive_threshold`, plus:
  - `max_exact_calculations` (`int`): Budget for exact MCES calculations (default `1000`).

- **Returns:**
  - `Tuple[list[str], list[str], list[str], int]`: `(train_set, validation_set, test_set, final_threshold)`

---

#### `split_dataset_umap(dataset, validation_fraction=0.1, test_fraction=0.1, min_ratio=0.7, random_state=42, mces_matrix_save_path=None, hdbscan_kwargs=None, **umap_kwargs)`

Splits a dataset by embedding the MCES lower-bound distance matrix with UMAP, clustering the embedding with HDBSCAN, and distributing clusters across splits. If HDBSCAN collapses to too few clusters (common on small or very homogeneous datasets), the embedding is re-partitioned with k-means so that validation and test sets can still be populated.

- **Parameters:**
  - `dataset` (`list[str]`): List of SMILES strings.
  - `validation_fraction` (`float`): Target fraction for validation set.
  - `test_fraction` (`float`): Target fraction for test set.
  - `min_ratio` (`float`): Minimum required size ratio for validation/test vs target.
  - `random_state` (`int`): Random seed for UMAP and cluster shuffling (default `42`).
  - `mces_matrix_save_path` (`str | None`): If provided, saves the lower-bound matrix to this path as a `.npy` file.
  - `hdbscan_kwargs` (`dict | None`): Optional keyword arguments passed to `hdbscan.HDBSCAN`.
  - `**umap_kwargs`: Keyword arguments passed to `umap.UMAP`. `metric` is fixed to `"precomputed"`. `random_state` defaults to the value of `random_state`. `n_neighbors` is capped at `n - 1`.

- **Returns:**
  - `Tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]`: `(train_set, validation_set, test_set, bounds_matrix, umap_embedding)`

---

#### `find_critical_pairs_for_threshold_optimization(dataset, bounds_matrix, current_threshold, validation_fraction=0.1, test_fraction=0.1, min_ratio=0.7, max_exact_calculations=1000)`

Identifies molecule pairs where calculating exact MCES is most likely to improve the current split threshold.

- **Parameters:**
  - `dataset` (`list[str]`): List of SMILES strings.
  - `bounds_matrix` (`np.ndarray`): Precomputed lower-bound distance matrix.
  - `current_threshold` (`int`): Current distinction threshold.
  - `validation_fraction`, `test_fraction`, `min_ratio`: Split constraints.
  - `max_exact_calculations` (`int`): Maximum pairs to return.

- **Returns:**
  - `list[tuple[int, int]]`: Sorted list of critical `(i, j)` pairs.

---

#### `try_split_with_threshold(dataset, distance_matrix, threshold, validation_fraction, test_fraction, min_ratio, random_seed=None)`

Attempts a single split with a given distance matrix and threshold. Returns `None` sets if constraints are not met.

- **Parameters:**
  - `dataset` (`list[str]`): List of SMILES strings.
  - `distance_matrix` (`np.ndarray`): Distance matrix (lower bounds or exact).
  - `threshold` (`int`): Distinction threshold.
  - `validation_fraction`, `test_fraction`, `min_ratio`: Split constraints.
  - `random_seed` (`int | None`): Optional random seed for reproducibility.

- **Returns:**
  - `Tuple[list[str], list[str], list[str]] | Tuple[None, None, None]`: The split or failure.

---

### `mces_splitting.bounds_test`

#### `bounds_validity_test(data_file_path, skip_mces=False)`

Runs a comprehensive test validating that lower bounds never exceed the true MCES distance. Compares `filter1`, `filter2_from_lib`, and the C++ `mces_lower_bound` implementation.

- **Parameters:**
  - `data_file_path` (`str`): Path to a CSV or dataset file.
  - `skip_mces` (`bool`): If `True`, skips exact MCES calculations (faster).

---

## Package-Level Convenience Exports

The following are imported directly from the top-level `mces_splitting` package:

| Export Name | Points To |
|-------------|-----------|
| `mces_lower_bounds` | `bounds.mces_lower_bound_symmetric` |
| `exact_mces_for_list_of_pairs` | `mces.exact_mces_for_list_of_pairs` |
| `split_dataset_lower_bound_only` | `dataset_splitting.split_dataset_adaptive_threshold` |
| `split_dataset_with_exact_mces` | `dataset_splitting.split_dataset_brute_force_exact` |
| `split_dataset_umap` | `dataset_splitting.split_dataset_umap` |
| `split_dataset` | `dataset_splitting.split_dataset` |

---

## Command-Line Interface

After installation, the package provides a `mces-split` command for splitting SMILES files from the terminal.

### Usage

```bash
mces-split molecules.smi --method threshold --output-dir splits/
mces-split molecules.smi --method umap --validation-fraction 0.1 --test-fraction 0.1 --output-dir splits/
```

### Input

The CLI reads `.smi` or `.txt` files containing one SMILES string per line. Empty lines and lines starting with `#` are ignored.

### Output

Three files are written to `--output-dir`:

- `<input_stem>_train.smi`
- `<input_stem>_val.smi`
- `<input_stem>_test.smi`

Use `--output-prefix` to override the file prefix.

### Options

- `--method {threshold,umap}` — Splitting strategy (default: `threshold`).
- `--validation-fraction` — Target fraction for validation set (default: `0.1`).
- `--test-fraction` — Target fraction for test set (default: `0.1`).
- `--min-ratio` — Minimum required size ratio for validation/test vs target (default: `0.7`).
- `--random-state` — Random seed (default: `42`).
- `--output-dir` — Directory for output files (default: current directory).
- `--output-prefix` — Prefix for output files (default: input file stem).
- `--mces-matrix-save-path` — Optional path to save the lower-bound distance matrix.

Threshold-specific options:

- `--initial-distinction-threshold` (default: `10`)
- `--min-distinction-threshold` (default: `2`)
- `--threshold-step` (default: `-1`)

UMAP-specific options:

- `--n-components` (default: `2`)
- `--n-neighbors` (default: capped at `n - 1`)
- `--min-dist` (default: `0.1`)
- `--hdbscan-min-cluster-size` (default: adaptive)
- `--hdbscan-min-samples` (default: `1`)

---

## How It Works

### MCES Distance
The Maximum Common Edge Subgraph (MCES) distance measures the structural difference between two molecules by finding the largest common subgraph and computing the sum of bond weights that differ.

### Lower Bounds
Two fast lower-bound filters are provided:
1. **`filter1`** — Degree-based bound (fast, loose).
2. **`filter2_from_lib`** — Neighborhood-based bound using minimum-weight matching (slower, tighter). The C++ `fast_mces_lower_bound` extension implements an optimized version of this bound.

### Dataset Splitting
1. Compute pairwise lower bounds (or exact distances).
2. Build a graph where edges connect molecules with distance `< threshold` (non-distinct).
3. Find connected components (clusters of similar molecules).
4. Distribute clusters into train/validation/test using a round-robin strategy:
   - Large clusters go to training.
   - Small clusters are assigned to validation or test to meet target fractions.
5. If constraints are not met, lower the threshold and retry.

### UMAP-Based Splitting
1. Compute the symmetric MCES lower-bound distance matrix.
2. Embed the molecules into low-dimensional space with UMAP using the lower-bound matrix as a precomputed distance matrix.
3. Cluster the UMAP embedding with HDBSCAN. Noise points are treated as singleton clusters.
4. If HDBSCAN produces too few clusters (common on small or homogeneous datasets), the embedding is re-partitioned with k-means.
5. Distribute clusters into train/validation/test so each split covers the structural diversity represented in the UMAP space.

---

## License

See the repository for licensing details.
