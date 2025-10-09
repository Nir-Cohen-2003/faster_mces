import os
import sys
import argparse
from time import perf_counter
from hrms_utils.rdkit import sanitize_smiles_polars
from typing import List, Tuple
import polars as pl
import numpy as np
from mces_splitting import (
    split_dataset_lower_bound_only,
    split_dataset_with_exact_mces
)


def test_matrix_validation(csv_path: str, nist_smiles: List[str]):
    """Test matrix validation scenarios."""
    print("\n" + "="*80)
    print("TEST 3: MATRIX VALIDATION")
    print("="*80)
    
    test_matrix_path = os.path.join(os.path.dirname(__file__), "__pycache__", "test_validation_matrix.npy")
    
    # Test 3.1: Calculate fresh matrix and save it
    print("\n--- Test 3.1: Calculate fresh matrix and save it ---")
    train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        mces_matrix_save_path=test_matrix_path
    )
    print(f"✓ Matrix saved to {test_matrix_path}")
    
    # Test 3.2: Use the saved matrix
    print("\n--- Test 3.2: Use the saved matrix ---")
    train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        use_saved_mces_matrix_path=test_matrix_path
    )
    print("✓ Successfully used saved matrix")
    
    # Test 3.3: Corrupt the matrix and verify warning
    print("\n--- Test 3.3: Corrupt matrix and verify warning ---")
    saved_matrix = np.load(test_matrix_path)
    corrupted_matrix = saved_matrix.copy()
    corrupted_matrix[0, 1] = corrupted_matrix[0, 1] + 999
    corrupted_matrix[1, 0] = corrupted_matrix[1, 0] + 999
    
    corrupted_path = os.path.join(os.path.dirname(__file__), "__pycache__", "corrupted_matrix.npy")
    np.save(corrupted_path, corrupted_matrix)
    
    print("Attempting to use corrupted matrix (should see warning)...")
    train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        use_saved_mces_matrix_path=corrupted_path
    )
    print("✓ Corrupted matrix was rejected and new calculation performed")
    
    # Cleanup
    if os.path.exists(test_matrix_path):
        os.remove(test_matrix_path)
    if os.path.exists(corrupted_path):
        os.remove(corrupted_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dataset splitting with optional row limit")
    parser.add_argument("n_rows", type=int, nargs="?", default=None, 
                       help="Number of rows to take from dataset (uses .head())")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip matrix validation tests")
    parser.add_argument("--skip-exact", action="store_true",
                       help="Skip brute force exact calculation test")
    args = parser.parse_args()

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pl.scan_csv(csv_path).select("MS_READY_SMILES").unique(maintain_order=True)
    
    if args.n_rows is not None:
        print(f"Taking first {args.n_rows} rows from dataset")
        df = df.head(args.n_rows)
    
    nist_smiles: List[str] = df.with_columns(
        pl.col("MS_READY_SMILES").map_batches(function=sanitize_smiles_polars, return_dtype=pl.String)
    ).filter(
        pl.col("MS_READY_SMILES").is_not_null(),
        pl.col("MS_READY_SMILES").ne("")
    ).collect().to_series().to_list()
    
    if any(smile == "=" for smile in nist_smiles):
        raise ValueError("Invalid SMILES found in dataset. Please check the input data.")
    
    os.makedirs(os.path.join(os.path.dirname(__file__), "__pycache__"), exist_ok=True)

    print(f"\n{'='*80}")
    print(f"DATASET SPLITTING TEST - {len(nist_smiles)} molecules")
    print(f"{'='*80}\n")

    # Test 1: Lower bound only (adaptive)
    print("="*80)
    print("TEST 1: ADAPTIVE (LOWER BOUND ONLY)")
    print("="*80)
    start = perf_counter()
    train_set, validation_set, test_set, threshold_adaptive = split_dataset_lower_bound_only(
        nist_smiles.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
    )
    end = perf_counter()
    adaptive_time = end - start

    print(f"\nTraining set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Total size: {len(nist_smiles)}")
    print(f"Threshold: {threshold_adaptive}")
    print(f"Time taken: {adaptive_time:.2f} seconds")

    # Test 2: Brute force exact calculation (optional)
    if not args.skip_exact:
        print(f"\n{'='*80}")
        print("TEST 2: BRUTE FORCE EXACT CALCULATION")
        print("="*80)
        start = perf_counter()
        train_set, validation_set, test_set, threshold_brute_force = split_dataset_with_exact_mces(
            nist_smiles.copy(),
            validation_fraction=0.1,
            test_fraction=0.1,
            initial_distinction_threshold=10,
            min_distinction_threshold=0,
            threshold_step=-1,
            max_exact_calculations=30_000,
        )
        end = perf_counter()
        brute_force_time = end - start

        print(f"\nTraining set size: {len(train_set)}")
        print(f"Validation set size: {len(validation_set)}")
        print(f"Test set size: {len(test_set)}")
        print(f"Total size: {len(nist_smiles)}")
        print(f"Threshold: {threshold_brute_force}")
        print(f"Time taken (brute force): {brute_force_time:.2f} seconds")
        
        # Comparison
        print(f"\n{'='*80}")
        print("COMPARISON")
        print("="*80)
        print(f"Adaptive threshold: {threshold_adaptive}")
        print(f"Brute force threshold: {threshold_brute_force}")
        print(f"Adaptive time: {adaptive_time:.2f} seconds")
        print(f"Brute force time: {brute_force_time:.2f} seconds")
        print(f"Brute force vs Adaptive slowdown: {brute_force_time / adaptive_time:.2f}x")
        print(f"Threshold improvement: {threshold_brute_force - threshold_adaptive}")
    else:
        print(f"\n(Skipping brute force exact calculation test)")

    # Test 3: Matrix validation (runs by default)
    if not args.skip_validation:
        test_matrix_validation(csv_path, nist_smiles)
    else:
        print(f"\n(Skipping matrix validation tests)")