import polars as pl
import fast_mces_lower_bound
from time import perf_counter
import os

N = 3000  # Number of SMILES to benchmark

# Read "dsstox_smiles_medium.csv" and extract first N SMILES
# next to this file
dsstox_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
df = pl.read_csv(dsstox_file_path)
smiles_list = df.select("MS_READY_SMILES").filter(
    pl.col("MS_READY_SMILES").is_not_null(),
    pl.col("MS_READY_SMILES") != "",
    ~pl.col("MS_READY_SMILES").str.contains("Sn",literal=True)
).head(N).to_series().to_list()

start = perf_counter()
result = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
end = perf_counter()

print(f"Calculated symmetric distance matrix for {N} SMILES in {end - start:.4f} seconds.")