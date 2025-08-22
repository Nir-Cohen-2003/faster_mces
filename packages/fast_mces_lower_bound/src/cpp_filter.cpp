#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "cpp_filter.hpp"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <algorithm>
#include <stdexcept>
#include <set>
#include "lap.h"
#include <chrono>
#include <atomic>

#ifndef FAST_MCES_PROFILE
#define FAST_MCES_PROFILE 0
#endif

#ifndef FAST_MCES_ERROR_COUNT
#define FAST_MCES_ERROR_COUNT 0
#endif

#if FAST_MCES_PROFILE
static std::atomic<long long> g_precompute_time_ns{0};
static std::atomic<long long> g_pair_fill_time_ns{0};
static std::atomic<long long> g_pair_solve_time_ns{0};
static std::atomic<long long> g_precompute_count{0};
static std::atomic<long long> g_pair_count{0};
#endif

cost solve_lap(const std::vector<cost>& cost_matrix_flat, size_t n) {
    // cost_matrix_flat is row-major: cost_matrix_flat[i * n + j]
    std::vector<cost*> cost_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        cost_ptrs[i] = const_cast<cost*>(&cost_matrix_flat[i * n]);
    }
    std::vector<int> rowsol(n), colsol(n);
    std::vector<cost> u(n), v(n);
    cost lap_cost = lap(n, cost_ptrs.data(), rowsol.data(), colsol.data(), u.data(), v.data());
    return lap_cost;
}


PrecomputedMol precompute_mol_data(const std::string& smiles) {
    std::unique_ptr<RDKit::ROMol> mol(RDKit::SmilesToMol(smiles));
    if (!mol) {
        throw std::runtime_error("Failed to parse SMILES: " + smiles);
    }

    PrecomputedMol p_mol;
    p_mol.atom_data_vec.resize(mol->getNumAtoms());

    for (const auto atom : mol->atoms()) {
        unsigned int atom_idx = atom->getIdx();
        int atom_type = atom->getAtomicNum();
        p_mol.atom_types_to_indices[atom_type].push_back(atom_idx);

        AtomData& atom_data = p_mol.atom_data_vec[atom_idx];
        atom_data.total_weight = 0.0;

        for (const auto& neighbor : mol->atomNeighbors(atom)) {
            const RDKit::Bond* bond = mol->getBondBetweenAtoms(atom_idx, neighbor->getIdx());
            cost weight = static_cast<cost>(bond->getBondTypeAsDouble());
            int neighbor_type = neighbor->getAtomicNum();
            atom_data.atom_weights[neighbor_type].push_back(weight);
            atom_data.total_weight += weight;
        }
        
        atom_data.total_weight /= 2.0;

        for (auto& pair : atom_data.atom_weights) {
            std::sort(pair.second.rbegin(), pair.second.rend());
        }
    }
    return p_mol;
}

static const std::vector<cost> kEmptyCostVec;

// Replace node_cost internals to avoid temporaries
// cost node_cost(unsigned int node1_idx, const PrecomputedMol& mol1, unsigned int node2_idx, const PrecomputedMol& mol2) {
//     const auto& weights1_map = mol1.atom_data_vec[node1_idx].atom_weights;
//     const auto& weights2_map = mol2.atom_data_vec[node2_idx].atom_weights;
//
//     std::set<int> all_atom_types;
//     for (const auto& pair : weights1_map) all_atom_types.insert(pair.first);
//     for (const auto& pair : weights2_map) all_atom_types.insert(pair.first);
//
//     cost cost_val = 0.0;
//     for (int atom_type : all_atom_types) {
//         auto it1 = weights1_map.find(atom_type);
//         auto it2 = weights2_map.find(atom_type);
//
//         const std::vector<cost>* w1 = (it1 != weights1_map.end()) ? &it1->second : &kEmptyCostVec;
//         const std::vector<cost>* w2 = (it2 != weights2_map.end()) ? &it2->second : &kEmptyCostVec;
//
//         size_t min_len = std::min(w1->size(), w2->size());
//         for (size_t i = 0; i < min_len; ++i) {
//             cost_val += std::abs((*w1)[i] - (*w2)[i]);
//         }
//
//         for (size_t i = min_len; i < w1->size(); ++i) cost_val += (*w1)[i];
//         for (size_t i = min_len; i < w2->size(); ++i) cost_val += (*w2)[i];
//     }
//     return cost_val / 2.0;
// }

// New helper: build dense flat features for one molecule given global ordering and per-type maxima
static void build_flat_features_for_mol(PrecomputedMol& pmol,
                                        const std::vector<int>& global_types,
                                        const std::vector<size_t>& max_counts_per_type) {
    size_t n_atoms = pmol.atom_data_vec.size();
    size_t per_type_total = 0;
    for (size_t c : max_counts_per_type) per_type_total += c;
    // optional: append one extra element for total_weight at the end
    size_t vec_len = per_type_total;
    pmol.flat_features.assign(n_atoms, std::vector<cost>(vec_len, 0.0));

    for (size_t a = 0; a < n_atoms; ++a) {
        std::vector<cost>& feat = pmol.flat_features[a];
        size_t offset = 0;
        for (size_t t_i = 0; t_i < global_types.size(); ++t_i) {
            int atype = global_types[t_i];
            auto it = pmol.atom_data_vec[a].atom_weights.find(atype);
            const std::vector<cost>* w = (it != pmol.atom_data_vec[a].atom_weights.end()) ? &it->second : &kEmptyCostVec;
            size_t maxc = max_counts_per_type[t_i];
            for (size_t k = 0; k < maxc; ++k) {
                feat[offset + k] = (k < w->size()) ? (*w)[k] : 0.0;
            }
            offset += maxc;
        }
    }
}

// New fast node cost using dense vectors (L1) - equivalent to old logic
static inline cost node_cost_flat(const std::vector<cost>& f1, const std::vector<cost>& f2) {
    // assume same length
    size_t L = f1.size();
    cost s = 0.0;
    for (size_t i = 0; i < L; ++i) s += std::abs(f1[i] - f2[i]);
    return s / 2.0;
}

// Modify calculate_symmetric_distance_matrix to build global types and flat features once per batch
std::vector<cost> calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list) {
    size_t n = smiles_list.size();
    std::vector<PrecomputedMol> precomputed_mols(n);

#if FAST_MCES_PROFILE
    auto t0_all = std::chrono::steady_clock::now();
#endif

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        precomputed_mols[i] = precompute_mol_data(smiles_list[i]);
    }

#if FAST_MCES_PROFILE
    auto t1_all = std::chrono::steady_clock::now();
    g_precompute_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1_all - t0_all).count());
    g_precompute_count.fetch_add(static_cast<long long>(n));
#endif

    // collect global atom types and compute max neighbor counts per type
    std::map<int, size_t> max_counts_map;
    for (const auto& pm : precomputed_mols) {
        for (size_t a = 0; a < pm.atom_data_vec.size(); ++a) {
            for (const auto& p : pm.atom_data_vec[a].atom_weights) {
                int atype = p.first;
                max_counts_map[atype] = std::max(max_counts_map[atype], p.second.size());
            }
        }
    }
    std::vector<int> global_types;
    std::vector<size_t> max_counts_per_type;
    global_types.reserve(max_counts_map.size());
    max_counts_per_type.reserve(max_counts_map.size());
    for (const auto& p : max_counts_map) {
        global_types.push_back(p.first);
        max_counts_per_type.push_back(p.second);
    }

    // build dense flat features per molecule (parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        build_flat_features_for_mol(precomputed_mols[i], global_types, max_counts_per_type);
    }

    // now use the faster flat-features when filling the cost blocks
    return filter2_batch_symmetric(precomputed_mols);
}

cost calculate_pair_distance(const PrecomputedMol& mol1, const PrecomputedMol& mol2) {
    static const std::vector<unsigned int> kEmptyNodes;

    #if FAST_MCES_PROFILE
    g_pair_count.fetch_add(1);
#endif

    cost total_cost = 0.0;
    std::set<int> all_types;
    for (const auto& p : mol1.atom_types_to_indices) all_types.insert(p.first);
    for (const auto& p : mol2.atom_types_to_indices) all_types.insert(p.first);

    for (int atom_type : all_types) {
        const auto it1 = mol1.atom_types_to_indices.find(atom_type);
        const auto it2 = mol2.atom_types_to_indices.find(atom_type);

        const std::vector<unsigned int>& nodes1 = (it1 != mol1.atom_types_to_indices.end()) ? it1->second : kEmptyNodes;
        const std::vector<unsigned int>& nodes2 = (it2 != mol2.atom_types_to_indices.end()) ? it2->second : kEmptyNodes;

        if (nodes1.empty()) {
            for (unsigned int n2 : nodes2) total_cost += mol2.atom_data_vec[n2].total_weight;
            continue;
        }
        if (nodes2.empty()) {
            for (unsigned int n1 : nodes1) total_cost += mol1.atom_data_vec[n1].total_weight;
            continue;
        }

        size_t n1 = nodes1.size();
        size_t n2 = nodes2.size();
        size_t m = std::max(n1, n2);
        std::vector<cost> cost_matrix(m * m, 0.0);

        // fill cost_matrix and time the fill if profiling enabled
        for (size_t i = 0; i < n1; ++i) {
#if FAST_MCES_PROFILE
            auto fill_start = std::chrono::steady_clock::now();
#endif
            for (size_t j = 0; j < n2; ++j) {
                // always use dense flat features
                cost_matrix[i * m + j] = node_cost_flat(mol1.flat_features[nodes1[i]],
                                                        mol2.flat_features[nodes2[j]]);
            }
#if FAST_MCES_PROFILE
            auto fill_end = std::chrono::steady_clock::now();
            g_pair_fill_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(fill_end - fill_start).count());
#endif
        }

        // pad remaining rows/cols with total_weight
#if FAST_MCES_PROFILE
        auto pad_start = std::chrono::steady_clock::now();
#endif
        if (n1 < m) {
            for (size_t i = n1; i < m; ++i)
                for (size_t j = 0; j < n2; ++j)
                    cost_matrix[i * m + j] = mol2.atom_data_vec[nodes2[j]].total_weight;
        }
        if (n2 < m) {
            for (size_t i = 0; i < n1; ++i)
                for (size_t j = n2; j < m; ++j)
                    cost_matrix[i * m + j] = mol1.atom_data_vec[nodes1[i]].total_weight;
        }
#if FAST_MCES_PROFILE
        auto pad_end = std::chrono::steady_clock::now();
        g_pair_fill_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(pad_end - pad_start).count());
#endif

        // solve LAP and time it if profiling enabled
#if FAST_MCES_PROFILE
        auto solve_start = std::chrono::steady_clock::now();
#endif
        total_cost += solve_lap(cost_matrix, m);
#if FAST_MCES_PROFILE
        auto solve_end = std::chrono::steady_clock::now();
        g_pair_solve_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count());
#endif
    }

    return total_cost;
}

std::vector<cost> filter2_batch_symmetric(const std::vector<PrecomputedMol>& mols) {
    size_t n = mols.size();
    std::vector<cost> results(n * n, 0.0);

#if FAST_MCES_ERROR_COUNT
    int error_count = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:error_count)
#else
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            if (i == j) {
                results[i * n + j] = 0.0;
                continue;
            }

            cost dist = calculate_pair_distance(mols[i], mols[j]);

#if FAST_MCES_ERROR_COUNT
            if (std::isfinite(dist) && dist >= 0 && dist <= 10000.0) {
                results[i * n + j] = dist;
                results[j * n + i] = dist;
            } else {
                results[i * n + j] = 0.0;
                results[j * n + i] = 0.0;
                error_count++;
            }
#else
            results[i * n + j] = dist;
            results[j * n + i] = dist;
#endif
        }
    }

#if FAST_MCES_ERROR_COUNT
    if (error_count > 0) {
        std::cerr << "Warning: " << error_count << " pairwise distances exceeded 10000." << std::endl;
    }
#endif

#if FAST_MCES_PROFILE
    long long pre_ns = g_precompute_time_ns.load();
    long long fill_ns = g_pair_fill_time_ns.load();
    long long solve_ns = g_pair_solve_time_ns.load();
    long long pre_count = g_precompute_count.load();
    long long pair_count = g_pair_count.load();
    std::cerr << "[PROFILE] precompute total: " << (pre_ns / 1e6) << " ms"
              << " (molecules: " << pre_count << ", avg per mol: " << (pre_count? (pre_ns / 1e6 / pre_count) : 0) << " ms)"
              << std::endl;
    std::cerr << "[PROFILE] pair fill total: " << (fill_ns / 1e6) << " ms"
              << " (pairs: " << pair_count << ", avg fill per pair: " << (pair_count? (fill_ns / 1e6 / pair_count) : 0) << " ms)"
              << std::endl;
    std::cerr << "[PROFILE] pair solve total: " << (solve_ns / 1e6) << " ms"
              << " (pairs: " << pair_count << ", avg solve per pair: " << (pair_count? (solve_ns / 1e6 / pair_count) : 0) << " ms)"
              << std::endl;
#endif

    return results;
}



// Modify calculate_distance_matrix similarly: after precomputing mols1 and mols2 build a global type list across both and then call build_flat_features_for_mol on all molecules in parallel.
std::vector<cost> calculate_distance_matrix(const std::vector<std::string>& smiles_list1, const std::vector<std::string>& smiles_list2) {
    size_t n1 = smiles_list1.size();
    size_t n2 = smiles_list2.size();
    std::vector<PrecomputedMol> mols1(n1);
    std::vector<PrecomputedMol> mols2(n2);

#if FAST_MCES_PROFILE
    auto t0 = std::chrono::steady_clock::now();
#endif
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n1; ++i) {
        mols1[i] = precompute_mol_data(smiles_list1[i]);
    }
#if FAST_MCES_PROFILE
    auto t1 = std::chrono::steady_clock::now();
    g_precompute_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    g_precompute_count.fetch_add(static_cast<long long>(n1));
#endif

#if FAST_MCES_PROFILE
    auto t2 = std::chrono::steady_clock::now();
#endif
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n2; ++j) {
        mols2[j] = precompute_mol_data(smiles_list2[j]);
    }
#if FAST_MCES_PROFILE
    auto t3 = std::chrono::steady_clock::now();
    g_precompute_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    g_precompute_count.fetch_add(static_cast<long long>(n2));
#endif

    std::vector<cost> results(n1 * n2, 0.0);

#if FAST_MCES_ERROR_COUNT
    int error_count = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+:error_count)
#else
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
#if FAST_MCES_ERROR_COUNT
            cost dist = calculate_pair_distance(mols1[i], mols2[j]);
            if (std::isfinite(dist) && dist >= 0 && dist <= 10000.0) {
                results[i * n2 + j] = dist;
            } else {
                results[i * n2 + j] = 0.0;
                error_count++;
            }
#else
            // When error counting is disabled, avoid the extra checks and assign directly.
            results[i * n2 + j] = calculate_pair_distance(mols1[i], mols2[j]);
#endif
        }
    }
#if FAST_MCES_ERROR_COUNT
    if (error_count > 0) {
        std::cerr << "Warning: " << error_count << " pairwise distances exceeded 10000." << std::endl;
    }
#endif

#if FAST_MCES_PROFILE
    long long pre_ns = g_precompute_time_ns.load();
    long long fill_ns = g_pair_fill_time_ns.load();
    long long solve_ns = g_pair_solve_time_ns.load();
    long long pre_count = g_precompute_count.load();
    long long pair_count = g_pair_count.load();
    std::cerr << "[PROFILE] precompute total: " << (pre_ns / 1e6) << " ms"
              << " (molecules: " << pre_count << ", avg per mol: " << (pre_count? (pre_ns / 1e6 / pre_count) : 0) << " ms)"
              << std::endl;
    std::cerr << "[PROFILE] pair fill total: " << (fill_ns / 1e6) << " ms"
              << " (pairs: " << pair_count << ", avg fill per pair: " << (pair_count? (fill_ns / 1e6 / pair_count) : 0) << " ms)"
              << std::endl;
    std::cerr << "[PROFILE] pair solve total: " << (solve_ns / 1e6) << " ms"
              << " (pairs: " << pair_count << ", avg solve per pair: " << (pair_count? (solve_ns / 1e6 / pair_count) : 0) << " ms)"
              << std::endl;
#endif

    return results;
}


