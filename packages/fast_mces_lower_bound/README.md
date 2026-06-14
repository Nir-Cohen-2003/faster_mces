# fast_mces_lower_bound

A high-performance C++ extension for computing **lower bounds** of the **Maximum Common Edge Subgraph (MCES)** distance between molecules given as SMILES strings.

This package uses:

- **RDKit** for SMILES parsing and molecular graph construction
- **OpenMP** for parallel processing
- **nanobind** for lightweight, zero-copy Python bindings
- **Boost** for serialization and iostreams

The lower bound is based on a neighborhood-aware matching cost (similar to `filter2` in the parent `mces_splitting` package) and is solved efficiently using the **Linear Assignment Problem (LAP)** algorithm.

---

## Installation

This package is built as a C++ extension module. It requires:

- CMake >= 3.15
- C++17 compiler (GCC, Clang, or MSVC)
- Python >= 3.8
- RDKit (headers and libraries)
- Boost (`system`, `serialization`, `iostreams`)
- OpenMP
- nanobind

### Build

```bash
cd packages/fast_mces_lower_bound
mkdir build && cd build
cmake .. -DRDKIT_INCLUDE_DIR=/path/to/rdkit/include \
         -DRDKIT_LIB_DIR=/path/to/rdkit/lib
make -j
make install
```

CMake options:

- `-DFAST_MCES_PROFILE=ON` — Enable timing/profiling code.
- `-DFAST_MCES_ERROR_COUNT=ON` — Enable error counting.

The `fast_mces_lower_bound` Python module is installed into the active Python environment's `site-packages`.

---

## Quick Start

```python
import fast_mces_lower_bound

smiles_list = [
    "CCO",      # ethanol
    "CC=O",     # acetaldehyde
    "CC",       # ethane
    "CCC",      # propane
    "c1ccccc1", # benzene
]

# Symmetric distance matrix (lower bounds)
sym_matrix = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
print(sym_matrix)

# Rectangular distance matrix between two lists
smiles_list2 = ["CCN", "C1CCCCC1"]
rect_matrix = fast_mces_lower_bound.calculate_distance_matrix(smiles_list, smiles_list2)
print(rect_matrix)
```

---

## Python API

### `fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)`

Computes a symmetric lower-bound distance matrix for a list of SMILES strings.

- **Parameters:**
  - `smiles_list` (`list[str]`): List of SMILES strings.

- **Returns:**
  - `np.ndarray`: A symmetric `(n, n)` NumPy array of type `float64` (or `double`), where `n = len(smiles_list)`.
    - `result[i, j]` is the lower-bound MCES distance between molecule `i` and molecule `j`.
    - The diagonal is `0`.
    - The matrix is symmetric: `result[i, j] == result[j, i]`.

The returned array is a **zero-copy** view of C++-allocated memory managed by a nanobind capsule.

---

### `fast_mces_lower_bound.calculate_distance_matrix(smiles_list1, smiles_list2)`

Computes a rectangular lower-bound distance matrix between two lists of SMILES strings.

- **Parameters:**
  - `smiles_list1` (`list[str]`): First list of SMILES strings.
  - `smiles_list2` (`list[str]`): Second list of SMILES strings.

- **Returns:**
  - `np.ndarray`: A `(n1, n2)` NumPy array where `n1 = len(smiles_list1)` and `n2 = len(smiles_list2)`.
    - `result[i, j]` is the lower-bound MCES distance between molecule `i` from `smiles_list1` and molecule `j` from `smiles_list2`.

The returned array is also a **zero-copy** view of C++-allocated memory.

---

## C++ Library API

The package also exposes a static C++ library (`mces_lower_bound`) that can be linked against directly.

### Header: `cpp_filter.hpp`

#### `PrecomputedMol`

Precomputed molecular data used to avoid re-parsing SMILES during repeated comparisons.

```cpp
struct PrecomputedMol {
    std::map<int, std::vector<unsigned int>> atom_types_to_indices;
    std::vector<AtomData> atom_data_vec;
    std::vector<std::vector<cost>> flat_features;
};
```

---

#### `AtomData`

Per-atom precomputed neighborhood information.

```cpp
struct AtomData {
    AtomWeightsMap atom_weights;   // map<atom_type, sorted_bond_weights>
    cost total_weight;             // sum of all incident bond weights
};
```

---

#### `precompute_mol_data(smiles) -> PrecomputedMol`

Parses a SMILES string with RDKit and precomputes all data needed for fast lower-bound distance calculation.

- **Parameters:**
  - `smiles` (`const std::string&`): SMILES string.

- **Returns:**
  - `PrecomputedMol`: Precomputed molecule data.

---

#### `calculate_pair_distance(mol1, mol2) -> cost`

Computes the lower-bound MCES distance between two precomputed molecules.

- **Parameters:**
  - `mol1`, `mol2` (`const PrecomputedMol&`): Precomputed molecule data.

- **Returns:**
  - `cost` (`double`): Lower-bound distance.

The algorithm works by:
1. Grouping atoms by element type.
2. For each atom type, building a cost matrix where each entry is the neighborhood matching cost between two atoms.
3. Solving the assignment problem (minimum-weight full matching) using the LAP solver.
4. Summing costs across all atom types.

---

#### `calculate_symmetric_distance_matrix(smiles_list) -> std::vector<cost>`

Computes all pairwise lower-bound distances for a list of SMILES strings and returns a flat vector in row-major order.

- **Parameters:**
  - `smiles_list` (`const std::vector<std::string>&`): List of SMILES strings.

- **Returns:**
  - `std::vector<cost>`: Flat `(n * n)` vector. Element at `i * n + j` corresponds to distance between `i` and `j`.

Internally uses **OpenMP** parallelization.

---

#### `calculate_distance_matrix(smiles_list1, smiles_list2) -> std::vector<cost>`

Computes all pairwise lower-bound distances between two lists of SMILES strings.

- **Parameters:**
  - `smiles_list1` (`const std::vector<std::string>&`): First list.
  - `smiles_list2` (`const std::vector<std::string>&`): Second list.

- **Returns:**
  - `std::vector<cost>`: Flat `(n1 * n2)` vector in row-major order.

---

#### `filter2_batch_symmetric(mols) -> std::vector<cost>`

Computes all pairwise lower-bound distances from a list of already-precomputed `PrecomputedMol` objects.

- **Parameters:**
  - `mols` (`const std::vector<PrecomputedMol>&`): Precomputed molecules.

- **Returns:**
  - `std::vector<cost>`: Flat symmetric distance matrix.

---

#### `solve_lap(cost_matrix_flat, n) -> cost`

Solves the Linear Assignment Problem for a square cost matrix given as a flat vector.

- **Parameters:**
  - `cost_matrix_flat` (`const std::vector<cost>&`): Flat `(n * n)` cost matrix.
  - `n` (`size_t`): Matrix dimension.

- **Returns:**
  - `cost`: Minimum total assignment cost.

---

### LAP Solver

The package includes a dedicated LAP (Linear Assignment Problem) solver optimized for dense matrices.

- **Header:** `lap.h`
- **Implementation:** `lap.cpp`

#### `cost`

The floating-point type used for costs. Typically `double`.

```cpp
using cost = double;  // defined in lap.h
```

---

## Performance Notes

- **Parallelism:** All batch matrix functions use OpenMP to parallelize over molecule pairs.
- **Zero-copy Python bindings:** The nanobind wrappers transfer ownership of C++-allocated arrays directly to NumPy without copying data.
- **Precomputation:** `PrecomputedMol` objects cache parsed molecular graphs and neighborhood features, making repeated comparisons against the same dataset very fast.

---

## Running Tests

```bash
cd packages/fast_mces_lower_bound
python tests/test.py
```

This will:
- Compute a symmetric distance matrix for a set of example SMILES.
- Verify the result is a NumPy array with correct shape.
- Assert non-negative values, symmetry, and zero diagonal.

---

## Running Benchmarks

```bash
cd packages/fast_mces_lower_bound
python tests/benchmark.py
```

This benchmarks the symmetric distance matrix calculation on a repeated set of 11 molecules (scaled to 11,000 total), and reports timing and peak RSS memory usage.

---

## Integration with `mces_splitting`

This package is used internally by `mces_splitting` for fast lower-bound matrix computation. The Python wrappers in `mces_splitting.bounds` (`mces_lower_bound_symmetric` and `mces_lower_bound`) simply call the C++ functions exposed here.

---

## License

See the repository for licensing details.
