import fast_mces_lower_bound
import numpy as np
# Example SMILES list
smiles_list = [
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


]

# Run symmetric distance matrix
sym_matrix = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
print("Symmetric distance matrix:", sym_matrix)
# print the type of what we got, and assert that it's a flattened list. print also the inner type
print("Type of sym_matrix:", type(sym_matrix))
if len(sym_matrix) > 0:
    print("Type of elements in sym_matrix:", type(sym_matrix[0]))
assert isinstance(sym_matrix, list), "sym_matrix is not a list"
assert all(isinstance(x, (int, float, np.integer, np.floating)) for x in sym_matrix), "sym_matrix contains non-numeric elements"
# make sure all the values


# now format it as a square numpy array
sym_matrix = np.array(sym_matrix, dtype=float).reshape((len(smiles_list), len(smiles_list)))
print("Formatted symmetric distance matrix: \n", sym_matrix)
# make sure all values are >= 0
assert np.all(sym_matrix >= 0), "Negative value found in symmetric distance matrix"
# print the maximal value
print("Max value in symmetric distance matrix:", np.max(sym_matrix))