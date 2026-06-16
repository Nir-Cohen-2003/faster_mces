"""Compare the C++ MCES upper bound against the top-level exact ILP MCES.

This test is intentionally standalone: it only uses the public API exposed by
``mces_splitting``:

* ``mces_distance_upper_bound`` (from the inner ``fast_mces_lower_bound`` C++
  package, re-exported at the top level).
* ``exact_mces_for_list_of_pairs`` (parallel ILP wrapper around ``MCES_ILP``).

Only the disconnected MCES mode is exposed/used, because the top-level ILP
solves MCES (not MCS/connected mode).

Both functions now report the *same* distance metric (the weighted ILP cost):

* The top-level ILP uses only the atom symbol and penalises bond-order
  differences. Its distance is the minimum total weight of unmapped bonds plus
  bond-order differences for mapped bonds.
* The C++ upper bound reports exactly that same metric for the clique it
  finds. Because the C++ compatibility is stricter than the ILP's (exact atom
  labels + exact bond type), every clique returned by the C++ code is a
  feasible ILP matching, and the weighted cost of that matching is therefore a
  valid upper bound on the ILP optimum.

The numpy "validation" here reports the distribution of ``upper - exact``;
negative values would mean the C++ heuristic underestimated the ILP distance,
which should not happen.
"""
import numpy as np
import pulp
from rdkit import Chem

from mces_splitting import exact_mces_for_list_of_pairs, mces_distance_upper_bound



SMILES_LIST = [
    "CCO",
    "CC=O",
    "CC",
    "CCC",
    "CCN",
    "c1ccccc1",
    "c1ccccc1C",
    "Nc1cc(C)ccc1",
    "Nc1ccc(C)cc1",
    "C1CCCCC1",
    "C1CCCCC1C",
    "CC(C)C",
    "CC(C)N",
    "[CH3-]",
    "C[C@H](O)N",
    "C[C@@H](O)N",
    "CC.CC",
    "CCCC",
]


def _upper_distance(s1: str, s2: str, num_starts: int = 200) -> float:
    """Disconnected MCES upper bound using the C++ clique heuristic.

    The C++ function reports the weighted ILP distance for the clique it
    finds, which is guaranteed to be an upper bound on the ILP distance.
    """
    res = mces_distance_upper_bound(
        s1, s2, {"connected": False, "num_starts": num_starts}
    )
    return float(res["distance_upper_bound"])


def _exact_distance_matrix(smiles_list: list[str]) -> np.ndarray:
    """All-to-all exact MCES distance matrix using the parallel ILP wrapper."""
    n = len(smiles_list)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # threshold=-1 removes the early-termination cut-off, so every pair gets the
    # true optimum from the ILP.  Disable CBC console spam for this run.
    pulp.LpSolverDefault.msg = False
    results = exact_mces_for_list_of_pairs(
        smiles_list,
        smiles_list,
        pairs,
        n_jobs=-1,
        batch_size=20,
        threshold=-1,
        solver="default",
    )

    mat = np.zeros((n, n), dtype=float)
    for i, j, dist in results:
        if dist is None:
            raise RuntimeError(f"Exact MCES failed for pair ({i}, {j})")
        mat[i, j] = mat[j, i] = float(dist)
    return mat


def _upper_distance_matrix(smiles_list: list[str]) -> np.ndarray:
    """All-to-all upper-bound distance matrix using the C++ heuristic."""
    n = len(smiles_list)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _upper_distance(smiles_list[i], smiles_list[j])
            mat[i, j] = mat[j, i] = d
    return mat


def _print_matrix(name: str, matrix: np.ndarray, labels: list[str]) -> None:
    print(f"\n{name} matrix ({len(labels)} x {len(labels)}):")
    width = max(len(l) for l in labels)
    header = " " * (width + 5) + " ".join(f"{i:>5}" for i in range(len(labels)))
    print(header)
    for i, lab in enumerate(labels):
        row = " ".join(f"{matrix[i, j]:>5.2f}" for j in range(len(labels)))
        print(f"{i:>2}: {lab:<{width}} {row}")


def _summarize(upper: np.ndarray, exact: np.ndarray, labels: list[str]) -> None:
    diff = upper - exact
    triu = np.triu_indices(len(labels), k=1)
    d = diff[triu]

    print("\n=== Gap distribution (upper - exact ILP, unique off-diagonal pairs) ===")
    print(f"pairs where upper == exact: {int(np.sum(np.abs(d) < 1e-9))}")
    print(f"pairs where upper > exact:  {int(np.sum(d > 1e-9))}")
    print(f"pairs where upper < exact:  {int(np.sum(d < -1e-9))}")
    print(f"mean(upper - exact) = {d.mean():.4f}")
    print(f"std(upper - exact)  = {d.std():.4f}")
    for p in [1, 10, 25, 50, 75, 90, 99]:
        print(f"  {p:2d}% percentile gap = {np.percentile(d, p):.2f}")

    if d.size > 0:
        pos_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            f"largest positive gap: index {pos_idx}  "
            f"{labels[pos_idx[0]]!r} vs {labels[pos_idx[1]]!r}, "
            f"gap = {diff[pos_idx]:.2f}"
        )
        neg_idx = np.unravel_index(np.argmin(diff), diff.shape)
        print(
            f"largest negative gap: index {neg_idx}  "
            f"{labels[neg_idx[0]]!r} vs {labels[neg_idx[1]]!r}, "
            f"gap = {diff[neg_idx]:.2f}"
        )


def main() -> None:
    # quick sanity check that every SMILES parses
    for s in SMILES_LIST:
        if Chem.MolFromSmiles(s) is None:
            raise ValueError(f"Failed to parse SMILES: {s!r}")

    labels = [f"{i}:{s}" for i, s in enumerate(SMILES_LIST)]

    exact_mat = _exact_distance_matrix(SMILES_LIST)
    upper_mat = _upper_distance_matrix(SMILES_LIST)

    _print_matrix("EXACT (ILP, disconnected MCES)", exact_mat, labels)
    _print_matrix("UPPER (C++ clique, disconnected MCES)", upper_mat, labels)
    _summarize(upper_mat, exact_mat, labels)


if __name__ == "__main__":
    main()
