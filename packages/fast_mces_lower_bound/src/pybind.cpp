#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "cpp_filter.hpp"
#include "mces_upper_bound.hpp"
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

static nb::dict py_mces_distance_upper_bound(const std::string& smiles1,
                                              const std::string& smiles2,
                                              nb::dict config) {
    bool connected = false;
    int num_starts = 100;

    if (config.contains("connected")) {
        connected = static_cast<bool>(nb::bool_(config["connected"]));
    }
    if (config.contains("num_starts")) {
        num_starts = static_cast<int>(nb::int_(config["num_starts"]));
    }

    McesUpperBoundResult res = mces_distance_upper_bound(smiles1, smiles2, connected, num_starts);

    nb::dict out;
    nb::list pairs;
    for (const auto& p : res.matched_edge_pairs) {
        pairs.append(nb::make_tuple(p.first, p.second));
    }
    out["matched_edge_pairs"] = pairs;
    out["matched_edge_count"] = res.matched_edge_count;
    out["distance_upper_bound"] = res.distance_upper_bound;
    out["association_vertex_count"] = res.association_vertex_count;
    out["association_edge_count"] = res.association_edge_count;
    out["runtime_ms"] = res.runtime_ms;

    nb::dict meta;
    meta["connected_mode"] = res.connected_mode;
    meta["compatibility_mode"] = res.compatibility_mode;
    meta["clique_heuristic"] = res.clique_heuristic;
    if (res.random_seed.has_value()) {
        meta["random_seed"] = *res.random_seed;
    } else {
        meta["random_seed"] = nb::none();
    }
    out["metadata"] = meta;

    return out;
}

NB_MODULE(fast_mces_lower_bound, m) {
    m.def("calculate_symmetric_distance_matrix", &py_calculate_symmetric_distance_matrix,
          "Calculate symmetric distance matrix from a list of SMILES strings");
    m.def("calculate_distance_matrix", &py_calculate_distance_matrix,
          "Calculate distance matrix from two lists of SMILES strings");
    m.def("mces_distance_upper_bound", &py_mces_distance_upper_bound,
          nb::arg("smiles1"), nb::arg("smiles2"), nb::arg("config") = nb::dict(),
          "Compute a deterministic clique-based upper bound on the MCES distance");
}
