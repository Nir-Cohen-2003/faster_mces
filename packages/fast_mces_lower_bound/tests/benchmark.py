import polars as pl
import fast_mces_lower_bound
from time import perf_counter
import os

N = 200  # Number of SMILES to benchmark

# Read "dsstox_smiles_medium.csv" and extract first N SMILES
# next to this file
smiles_list = [
    "C1CCCCC1", # cyclohexane
    "c1ccccc1", # benzene
    "c1ccc2cc3ccccc3cc2c1", # anthracene
    # antracene, but with the 2 middle carbon alipahtic
    "c1ccc2cc3cCCcc3cc2c1", # anthracene with two middle carbons aliphatic
]*N

start = perf_counter()
result = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
end = perf_counter()

print(f"Calculated symmetric distance matrix for {N*4} SMILES in {end - start:.4f} seconds.")