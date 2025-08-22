import polars as pl
import fast_mces_lower_bound
from time import perf_counter
import os

smiles_list = [
    "c1ccc2cc3ccccc3cc2c1", # anthracene
    "c1ccc2cc3cCCcc3cc2c1", # anthracene with two middle carbons aliphatic
    "CC(NC)CC1=CC=C(OCO2)C2=C1", # mdma
    "CCCCCc1cc(c2c(c1)OC([C@H]3[C@H]2C=C(CC3)C)(C)C)O", #THC
    r"Oc1c(c(O)cc(c1)CCCCC)[C@@H]2\C=C(/CC[C@H]2\C(=C)C)C", #CBD
    "O=C4[C@@H]5Oc1c2c(ccc1OC)C[C@H]3N(CC[C@]25[C@@]3(O)CC4)C", # oxycodone
    "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", # morphine
]*1000

start = perf_counter()
result = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
end = perf_counter()

print(f"Calculated symmetric distance matrix for {len(smiles_list)} SMILES in {end - start:.4f} seconds.")