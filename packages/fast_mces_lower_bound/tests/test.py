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
    # antracene, but with the 2 middle carbon aliphatic
    "c1ccc2cc3cCCcc3cc2c1", # anthracene with two middle carbons aliphatic
    "Nc1cc(C)ccc1",  # m-methylaniline (meta-methyl aniline)
    "Nc1ccc(C)cc1",  # p-methylaniline (para-methyl aniline)
    "CN(C)C(Cc1ccccc1)",  # methamphetamine (N-methylamphetamine)
    "CC(Cc1ccc(C)cc1)N",  # 4-methylamphetamine (methyl on the benzene ring, para)
]

# Run symmetric distance matrix
sym_matrix = fast_mces_lower_bound.calculate_symmetric_distance_matrix(smiles_list)
print("Symmetric distance matrix:", sym_matrix)

# Check that it's a numpy array (zero-copy from C++)
print("Type of sym_matrix:", type(sym_matrix))
print("Shape of sym_matrix:", sym_matrix.shape)
print("Dtype of sym_matrix:", sym_matrix.dtype)

assert isinstance(sym_matrix, np.ndarray), "sym_matrix is not a numpy array"
assert sym_matrix.shape == (len(smiles_list), len(smiles_list)), "sym_matrix has incorrect shape"

print("Formatted symmetric distance matrix: \n", sym_matrix)

# Make sure all values are >= 0
assert np.all(sym_matrix >= 0), "Negative value found in symmetric distance matrix"

# Print the maximal value
print("Max value in symmetric distance matrix:", np.max(sym_matrix))

# Check symmetry
assert np.allclose(sym_matrix, sym_matrix.T), "Matrix is not symmetric"

# Check diagonal is zero
assert np.allclose(np.diag(sym_matrix), 0), "Diagonal is not zero"

print("\nAll tests passed!")
