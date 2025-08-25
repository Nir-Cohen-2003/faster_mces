from fast_mces_lower_bound import calculate_distance_matrix, calculate_symmetric_distance_matrix
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

print(np.array([1]))

# Run symmetric distance matrix
sym_matrix = calculate_symmetric_distance_matrix(smiles_list)
print("Symmetric distance matrix:", sym_matrix)
# print the type of what we got, and assert that it's a flattened list. print also the inner type
print("Type of sym_matrix:", type(sym_matrix))
if len(sym_matrix) > 0:
    print("Type of elements in sym_matrix:", type(sym_matrix[0]))
assert isinstance(sym_matrix, list), "sym_matrix is not a list"
assert all(isinstance(x, (float)) for x in sym_matrix), "sym_matrix contains non-numeric elements"
# make sure all the values


# now format it as a square numpy array
sym_matrix = np.array(sym_matrix, dtype=float).reshape((len(smiles_list), len(smiles_list)))
print("Formatted symmetric distance matrix: \n", sym_matrix)
# make sure all values are >= 0
assert np.all(sym_matrix >= 0), "Negative value found in symmetric distance matrix"
# print the maximal value
print("Max value in symmetric distance matrix:", np.max(sym_matrix))

# Run non-symmetric distance matrix
matrix = calculate_distance_matrix(smiles_list,smiles_list)
print("Non-symmetric distance matrix:", matrix)
# print the type of what we got, and assert that it's a flattened list. print also the inner type
print("Type of matrix:", type(matrix))
if len(matrix) > 0:
    print("Type of elements in matrix:", type(matrix[0]))
assert isinstance(matrix, list), "matrix is not a list"
assert all(isinstance(x, (float)) for x in matrix), "matrix contains non-numeric elements"

# now format it as a square numpy array
matrix = np.array(matrix, dtype=float).reshape((len(smiles_list), len(smiles_list)))
print("Formatted non-symmetric distance matrix: \n", matrix)
# make sure all values are >= 0
assert np.all(matrix >= 0), "Negative value found in non-symmetric distance matrix"
# print the maximal value
print("Max value in non-symmetric distance matrix:", np.max(matrix))
