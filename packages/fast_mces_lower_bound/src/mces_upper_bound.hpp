#ifndef MCES_UPPER_BOUND_HPP
#define MCES_UPPER_BOUND_HPP

#include <string>
#include <vector>
#include <utility>
#include <optional>

struct McesUpperBoundResult {
    std::vector<std::pair<int, int>> matched_edge_pairs;
    int matched_edge_count = 0;
    double distance_upper_bound = 0.0;
    int association_vertex_count = 0;
    int association_edge_count = 0;
    int runtime_ms = 0;
    bool connected_mode = false;
    std::string compatibility_mode;
    std::string clique_heuristic;
    std::optional<int> random_seed;
};

McesUpperBoundResult mces_distance_upper_bound(const std::string& smiles1,
                                                 const std::string& smiles2,
                                                 bool connected = false,
                                                 int num_starts = 100);

std::vector<double> upper_bound_symmetric_matrix(const std::vector<std::string>& smiles_list,
                                                  bool connected = false,
                                                  int num_starts = 100);

std::vector<double> upper_bound_matrix(const std::vector<std::string>& smiles_list1,
                                        const std::vector<std::string>& smiles_list2,
                                        bool connected = false,
                                        int num_starts = 100);

#endif // MCES_UPPER_BOUND_HPP
