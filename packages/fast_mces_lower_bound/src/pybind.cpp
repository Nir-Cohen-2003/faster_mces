#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "cpp_filter.hpp"

namespace nb = nanobind;

std::vector<double> py_calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list) {
    return calculate_symmetric_distance_matrix(smiles_list);
}

std::vector<double> py_calculate_distance_matrix(const std::vector<std::string>& smiles_list1, const std::vector<std::string>& smiles_list2) {
    return calculate_distance_matrix(smiles_list1, smiles_list2);
}

NB_MODULE(fast_mces_lower_bound, m) {
    m.def("calculate_symmetric_distance_matrix", &py_calculate_symmetric_distance_matrix,
          "Calculate symmetric distance matrix from a list of SMILES strings");
    m.def("calculate_distance_matrix", &py_calculate_distance_matrix,
          "Calculate distance matrix from two lists of SMILES strings");
}