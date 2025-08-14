import fast_mces_lower_bound

# Example SMILES list
smiles_list = [
    "CCO",      # ethanol
    "CC",       # ethane
    "CCC",      # propane
    "CCN",      # ethylamine
    "C1CCCCC1", # cyclohexane
    "c1ccccc1", # benzene
]

# Run symmetric distance matrix
sym_matrix = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
print("Symmetric distance matrix:", sym_matrix)

# Run distance matrix with two lists (use smiles_list twice)
dist_matrix = fast_mces_lower_bound.calculate_distance_matrix(smiles_list, smiles_list*2)
print("Distance matrix:", dist_matrix)

# Check all values are >= 0
assert all(x >= 0 for x in sym_matrix), "Negative value found in symmetric distance matrix"
assert all(x >= 0 for x in dist_matrix), "Negative value found in distance matrix"

# Print maximum values
print("Max value in symmetric distance matrix:", max(sym_matrix))
print("Max value in distance matrix:", max(dist_matrix))