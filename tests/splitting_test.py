import os
from time import perf_counter
from hrms_utils.rdkit import sanitize_smiles_polars
from typing import List, Tuple
import polars as pl
from mces_splitting import (
    split_dataset_lower_bound_only,
    split_dataset_with_exact_mces
)


if __name__ == "__main__":
    # Load DSSTox CSV next to the src directory (one level up from this file)
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Read the single-column CSV ("MS_READY_SMILES"), sanitize and filter empty entries
    nist_smiles: List[str] = pl.scan_csv(csv_path).select("MS_READY_SMILES").unique(maintain_order=True).with_columns(
        pl.col("MS_READY_SMILES").map_batches(function=sanitize_smiles_polars, return_dtype=pl.String)
    ).filter(
        pl.col("MS_READY_SMILES").is_not_null(),
        pl.col("MS_READY_SMILES").ne("")
    ).collect().to_series().to_list()
    if any(smile == "=" for smile in nist_smiles):
        raise ValueError("Invalid SMILES found in dataset. Please check the input data.")
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "__pycache__"), exist_ok=True)

    start = perf_counter()
    train_set, validation_set, test_set, threshold_adaptive = split_dataset_lower_bound_only(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        # mces_matrix_save_path=os.path.join(os.path.dirname(__file__), "__pycache__", "mces_matrix.npy")
        
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    adaptive_time = end - start
    # now write the sets to files named train_set.parquet, validation_set.parquet, test_set.parquet
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set.parquet"))
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken: {adaptive_time:.2f} seconds")

    # # now use the selective exact calculation method
    # start = perf_counter()
    # train_set, validation_set, test_set, threshold_selective_exact = split_dataset_with_selective_exact_calculation(
    #     nist_smiles.copy(),
    #     validation_fraction=0.1,
    #     test_fraction=0.1,
    #     initial_distinction_threshold=10,
    #     min_distinction_threshold=0,
    #     threshold_step=-1,
    #     max_exact_calculations=10_000,
    # )  # type: Tuple[List[str], List[str], List[str], int]
    # end = perf_counter()
    # selective_time = end - start

    # # # now write them to similar files, but add _with_exact to the names
    # # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set_with_exact.parquet"))
    # # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set_with_exact.parquet"))
    # # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set_with_exact.parquet"))

    # print(f"Training set size: {len(train_set)}")
    # print(f"Validation set size: {len(validation_set)}")
    # print(f"Test set size: {len(test_set)}")
    # print(f"Total size: {len(nist_smiles)}")
    # print(f"Time taken (selective): {selective_time:.2f} seconds")
    
    # print(f"Adaptive threshold: {threshold}")
    # print(f"Selective exact calculation threshold: {threshold}")
    # print(f"Adaptive time: {adaptive_time:.2f} seconds")
    # print(f"Selective exact calculation time: {selective_time:.2f} seconds")
    # print(f"Speedup of forgoing exact calculations: {selective_time / adaptive_time:.2f}x")


    # now use the brute force exact calculation method
    start = perf_counter()
    train_set, validation_set, test_set, threshold_brute_force = split_dataset_with_exact_mces(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        max_exact_calculations=30_000,
    )  # type: Tuple[List[str], List[str], List[str], int]
    end = perf_counter()
    brute_force_time = end - start

    # # now write them to similar files, but add _brute_force to the names
    # pl.DataFrame({"CanonicalSMILES": train_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "train_set_brute_force.parquet"))
    # pl.DataFrame({"CanonicalSMILES": validation_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "validation_set_brute_force.parquet"))
    # pl.DataFrame({"CanonicalSMILES": test_set}).write_parquet(os.path.join(os.path.dirname(__file__), "__pycache__", "test_set_brute_force.parquet"))

    print(f"\n=== BRUTE FORCE EXACT RESULTS ===")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Time taken (brute force): {brute_force_time:.2f} seconds")
    
    print(f"\n=== COMPARISON ===")
    print(f"Adaptive threshold: {threshold_adaptive}")
    # print(f"Selective exact calculation threshold: {threshold_selective_exact}")
    print(f"Brute force exact calculation threshold: {threshold_brute_force}")
    print(f"Adaptive time: {adaptive_time:.2f} seconds")
    # print(f"Selective exact calculation time: {selective_time:.2f} seconds")
    print(f"Brute force exact calculation time: {brute_force_time:.2f} seconds")
    # print(f"Selective vs Adaptive speedup: {selective_time / adaptive_time:.2f}x")
    print(f"Brute force vs Adaptive is lower by: {brute_force_time / adaptive_time:.2f}x")
    # print(f"Selective vs Brute force speedup: {selective_time / brute_force_time:.2f}x")