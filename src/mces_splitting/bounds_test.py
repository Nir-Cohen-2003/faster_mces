import sys
import os
from time import perf_counter
from typing import List
import numpy as np
import polars as pl

# relative imports from the package
from .bounds import filter1, filter2_from_lib, mces_lower_bound_symmetric
from .mces import construct_graph, MCES_ILP, suppress_output


def bounds_validity_test(data_file_path: str, skip_mces: bool = False):
    logs: List[str] = []
    logs.append(f"Running bounds validity test on some example molecules, skip_mces={skip_mces}")
    # smiles_examples = pl.scan_csv(data_file_path).head(number_of_mol).collect().to_series().to_list()
    smiles_examples = [
    "CCO",      # ethanol
    "CC=O",
    "CC",       # ethane
    "CCC",      # propane
    "CCN",      # ethylamine
    "C1CCCCC1", # cyclohexane
    "c1ccccc1", # benzene
    "c1ccc2cc3ccccc3cc2c1", # anthracene
    # antracene, but with the 2 middle carbon alipahtic
    "c1ccc2cc3cCCcc3cc2c1", # anthracene with two middle carbons aliphatic
    "CC(NC)CC1=CC=C(OCO2)C2=C1", # mdma
    "CCCCCc1cc(c2c(c1)OC([C@H]3[C@H]2C=C(CC3)C)(C)C)O", #THC
    r"Oc1c(c(O)cc(c1)CCCCC)[C@@H]2\C=C(/CC[C@H]2\C(=C)C)C", #CBD
    "O=C4[C@@H]5Oc1c2c(ccc1OC)C[C@H]3N(CC[C@]25[C@@]3(O)CC4)C", # oxycodone
    "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", # morphine
    ]
    graphs = [construct_graph(smiles) for smiles in smiles_examples]

    # Assume the C++ implementation is available for this test (intentional)
    start_time = perf_counter()
    cpp_matrix = mces_lower_bound_symmetric(smiles_examples)
    mces_cpp_flat = list(np.array(cpp_matrix).ravel())
    time2_cpp = perf_counter() - start_time
    cpp_available = True
    logs.append("C++ mces lower-bound implementation used for comparison")

    # time filter1
    start_time = perf_counter()
    filter1_results = [filter1(G1, G2) for G1 in graphs for G2 in graphs]
    time1 = perf_counter() - start_time

    # time filter2 from lib
    start_time = perf_counter()
    filter2_from_lib_results = [filter2_from_lib(G1, G2) for G1 in graphs for G2 in graphs]
    time2_from_lib = perf_counter() - start_time

    # Compare the three bounds: filter1, filter2_from_lib, and C++ mces lower bound
    n_pairs = len(filter1_results)
    assert n_pairs == len(filter2_from_lib_results)
    assert cpp_available and n_pairs == len(mces_cpp_flat)

    eps = 1e-9
    f1_gt_f2 = [(idx, f1, f2) for idx, (f1, f2) in enumerate(zip(filter1_results, filter2_from_lib_results)) if f1 > f2 + eps]
    f2_gt_cpp = [(idx, f2, mc) for idx, (f2, mc) in enumerate(zip(filter2_from_lib_results, mces_cpp_flat)) if f2 > mc + eps]

    # NEW: ensure filter2_from_lib equals the C++ lower-bound (within eps)
    mismatches_f2_cpp = [(idx, f2, mc) for idx, (f2, mc) in enumerate(zip(filter2_from_lib_results, mces_cpp_flat)) if abs(f2 - mc) > eps]

    logs.append("\nComparison summary:")
    logs.append(f"Total pairs compared: {n_pairs}")
    logs.append(f"filter1 > filter2_from_lib: {len(f1_gt_f2)} pairs")
    if len(f1_gt_f2) > 0:
        logs.append("First 5 examples where filter1 > filter2_from_lib:")
        for t in f1_gt_f2[:5]:
            logs.append(str(t))
    logs.append(f"filter2_from_lib > mces_lower_bound (C++): {len(f2_gt_cpp)} pairs")
    if len(f2_gt_cpp) > 0:
        logs.append("First 5 examples where filter2_from_lib > mces_lower_bound:")
        for t in f2_gt_cpp[:5]:
            logs.append(str(t))

    # Report any exact-equality mismatches between filter2_from_lib and C++ mces_lower_bound
    logs.append(f"filter2_from_lib != mces_lower_bound (C++) (abs diff > {eps}): {len(mismatches_f2_cpp)} pairs")
    if len(mismatches_f2_cpp) > 0:
        logs.append("First 5 mismatches (idx, filter2_from_lib, c++_mces):")
        for t in mismatches_f2_cpp[:5]:
            logs.append(str(t))
        # Delay raising until after printing all logs at the end

    # Compute true MCES distances (slow) - skip if MCES_ILP unavailable or requested
    if not skip_mces:
        start_time = perf_counter()
        mces_results = []
        with suppress_output():
            for i, G1 in enumerate(graphs):
                for j, G2 in enumerate(graphs):
                    if i == j:
                        mces_results.append(0.0)
                    else:
                        try:
                            distance, distance_type = MCES_ILP(G1, G2, threshold=100, no_ilp_threshold=True, solver="default")
                            mces_results.append(distance)
                        except Exception as e:
                            logs.append(f"MCES_ILP failed for graphs {i}, {j}: {e}")
                            mces_results.append(float('inf'))
        time_mces = perf_counter() - start_time

        # Validate that none of the bounds exceed the true MCES distance
        invalid_bounds_found = False
        for idx, (f1, f2, true_d) in enumerate(zip(filter1_results, filter2_from_lib_results, mces_results)):
            if true_d == float('inf'):
                continue
            if f1 > true_d + eps:
                logs.append(f"INVALID BOUND: filter1 at idx {idx} = {f1} > true MCES = {true_d}")
                invalid_bounds_found = True
            if f2 > true_d + eps:
                logs.append(f"INVALID BOUND: filter2_from_lib at idx {idx} = {f2} > true MCES = {true_d}")
                invalid_bounds_found = True
            mc = mces_cpp_flat[idx]
            if mc > true_d + eps:
                logs.append(f"INVALID BOUND: mces_lower_bound (C++) at idx {idx} = {mc} > true MCES = {true_d}")
                invalid_bounds_found = True

        if not invalid_bounds_found:
            logs.append("\n✓ All bounds are valid (no filter exceeded true MCES distance)")
        else:
            logs.append("\n⚠️  INVALID BOUNDS DETECTED! See messages above.")
    else:
        if MCES_ILP is None:
            logs.append("MCES_ILP not available; skipping exact MCES calculations")
        else:
            logs.append("Skipping MCES calculation (skip_mces=True)")
        mces_results = None
        time_mces = 0

    # Print timing summary (also collected into logs to ensure all prints at end)
    logs.append("\nTiming results:")
    logs.append(f"Time for filter1: {time1:.2f} seconds")
    logs.append(f"Time for filter2_from_lib: {time2_from_lib:.2f} seconds")
    logs.append(f"Time for mces_lower_bound (C++): {time2_cpp:.2f} seconds")
    if mces_results is not None:
        logs.append(f"Time for MCES_ILP (true): {time_mces:.2f} seconds")

    # Print average differences between filters and MCES (mces_results if available, else C++ mces)
    if mces_results is not None:
        diffs_f1 = []
        diffs_f2 = []
        diffs_cpp = []
        for idx, (f1, f2, true_d) in enumerate(zip(filter1_results, filter2_from_lib_results, mces_results)):
            if true_d == float('inf'):
                continue
            diffs_f1.append(abs(f1 - true_d))
            diffs_f2.append(abs(f2 - true_d))
            diffs_cpp.append(abs(mces_cpp_flat[idx] - true_d))

        if len(diffs_f1) > 0:
            logs.append("\nAverage absolute differences to true MCES:")
            logs.append(f"avg |filter1 - MCES| = {np.mean(diffs_f1):.6f}")
            logs.append(f"avg |filter2_from_lib - MCES| = {np.mean(diffs_f2):.6f}")
            logs.append(f"avg |mces_lower_bound (C++) - MCES| = {np.mean(diffs_cpp):.6f}")
        else:
            logs.append("\nNo finite MCES results to compute average differences.")
    else:
        # fall back to comparing filters against the C++ lower-bound
        diffs_f1 = [abs(f1 - mc) for f1, mc in zip(filter1_results, mces_cpp_flat)]
        diffs_f2 = [abs(f2 - mc) for f2, mc in zip(filter2_from_lib_results, mces_cpp_flat)]
        logs.append("\nAverage absolute differences to mces_lower_bound (C++) reference:")
        logs.append(f"avg |filter1 - C++ mces| = {np.mean(diffs_f1):.6f}")
        logs.append(f"avg |filter2_from_lib - C++ mces| = {np.mean(diffs_f2):.6f}")

    # Emit all collected logs now
    for line in logs:
        print(line)

    # If the filter2 vs C++ mismatches exist, fail now (after printing)
    if len(mismatches_f2_cpp) > 0:
        raise AssertionError("filter2_from_lib and mces_lower_bound (C++) differ for some pairs")


if __name__ == "__main__":
    # simple CLI to call the validity test; keeps argument parsing minimal
    data_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
    skip_mces = '--no-mces' in sys.argv
    # allow overriding number via arg after the flag
    if len(sys.argv) > 1:
        try:
            for a in sys.argv[1:]:
                if a.isdigit():
                    number_of_mol = int(a)
                    break
        except Exception:
            pass
    bounds_validity_test(data_file_path, skip_mces=skip_mces)