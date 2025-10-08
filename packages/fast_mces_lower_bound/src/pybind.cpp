#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "cpp_filter.hpp"
#include "lap.h"
namespace nb = nanobind;

nb::ndarray<nb::numpy, cost> py_calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list) {
    auto result_vec = calculate_symmetric_distance_matrix(smiles_list);
    size_t n = smiles_list.size();
    
    // Transfer ownership to NumPy - zero copy
    cost* data = new cost[result_vec.size()];
    std::copy(result_vec.begin(), result_vec.end(), data);
    
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<cost*>(p);
    });
    
    return nb::ndarray<nb::numpy, cost>(data, {n, n}, owner);
}

nb::ndarray<nb::numpy, cost> py_calculate_distance_matrix(const std::vector<std::string>& smiles_list1, const std::vector<std::string>& smiles_list2) {
    auto result_vec = calculate_distance_matrix(smiles_list1, smiles_list2);
    size_t n1 = smiles_list1.size();
    size_t n2 = smiles_list2.size();
    
    cost* data = new cost[result_vec.size()];
    std::copy(result_vec.begin(), result_vec.end(), data);
    
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<cost*>(p);
    });
    
    return nb::ndarray<nb::numpy, cost>(data, {n1, n2}, owner);
}

NB_MODULE(fast_mces_lower_bound, m) {
    m.def("calculate_symmetric_distance_matrix", &py_calculate_symmetric_distance_matrix,
          "Calculate symmetric distance matrix from a list of SMILES strings");
    m.def("calculate_distance_matrix", &py_calculate_distance_matrix,
          "Calculate distance matrix from two lists of SMILES strings");
}