#ifndef CPP_FILTER_HPP
#define CPP_FILTER_HPP

#include <vector>
#include <string>
#include <map>
#include <GraphMol/ROMol.h>
#include "lap.h"
// A map from an atom type (int) to a list of bond weights (doubles).
// The list of weights is pre-sorted in descending order.
using AtomWeightsMap = std::map<int, std::vector<cost>>;

// Data for a single atom (node) needed for the cost calculation.
struct AtomData {
    AtomWeightsMap atom_weights;
    cost total_weight;
};

// Holds all pre-computed information for a single molecule.
struct PrecomputedMol {
    // Maps an atom type (int) to a list of atom indices of that type.
    std::map<int, std::vector<unsigned int>> atom_types_to_indices;
    // A vector of data for each atom, indexed by the atom's original index.
    std::vector<AtomData> atom_data_vec;
};

cost solve_lap(const std::vector<std::vector<cost>>& cost_matrix);

// This will be the single entry point from Cython.
std::vector<cost> calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list);

// New function: computes lower bound MCES matrix between two lists (not symmetric)
std::vector<cost> calculate_distance_matrix(const std::vector<std::string>& smiles_list1, const std::vector<std::string>& smiles_list2);

std::vector<cost> filter2_batch_symmetric(const std::vector<PrecomputedMol>& mols);

// Helper function to create a PrecomputedMol from a SMILES string.
// This will be called from Cython.
PrecomputedMol precompute_mol_data(const std::string& smiles);

#endif // CPP_FILTER_HPP