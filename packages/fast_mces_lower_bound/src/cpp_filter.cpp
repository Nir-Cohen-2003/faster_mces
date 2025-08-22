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

cost node_cost(unsigned int node1_idx, const PrecomputedMol& mol1, unsigned int node2_idx, const PrecomputedMol& mol2) {
    const auto& weights1_map = mol1.atom_data_vec[node1_idx].atom_weights;
    const auto& weights2_map = mol2.atom_data_vec[node2_idx].atom_weights;

    std::set<int> all_atom_types;
    for (const auto& pair : weights1_map) all_atom_types.insert(pair.first);
    for (const auto& pair : weights2_map) all_atom_types.insert(pair.first);

    cost cost_val = 0.0;
    for (int atom_type : all_atom_types) {
        auto it1 = weights1_map.find(atom_type);
        auto it2 = weights2_map.find(atom_type);

        const std::vector<cost>& w1 = (it1 != weights1_map.end()) ? it1->second : std::vector<cost>();
        const std::vector<cost>& w2 = (it2 != weights2_map.end()) ? it2->second : std::vector<cost>();

        size_t min_len = std::min(w1.size(), w2.size());
        for (size_t i = 0; i < min_len; ++i) {
            cost_val += std::abs(w1[i] - w2[i]);
        }
        
        for (size_t i = min_len; i < w1.size(); ++i) cost_val += w1[i];
        for (size_t i = min_len; i < w2.size(); ++i) cost_val += w2[i];
    }
    return cost_val / 2.0;
}

cost calculate_pair_distance(const PrecomputedMol& mol1, const PrecomputedMol& mol2) {
    cost total_cost = 0.0;
    
    std::set<int> all_types;
    for (const auto& pair : mol1.atom_types_to_indices) all_types.insert(pair.first);
    for (const auto& pair : mol2.atom_types_to_indices) all_types.insert(pair.first);

    // Static empty vectors to avoid repeated allocations
    static const std::vector<unsigned int> empty_nodes_vec;

#if FAST_MCES_PROFILE
    long long local_fill_ns = 0;
    long long local_solve_ns = 0;
#endif

    for (int atom_type : all_types) {
        auto it1 = mol1.atom_types_to_indices.find(atom_type);
        auto it2 = mol2.atom_types_to_indices.find(atom_type);

        const std::vector<unsigned int>& nodes1 = (it1 != mol1.atom_types_to_indices.end()) ? it1->second : empty_nodes_vec;
        const std::vector<unsigned int>& nodes2 = (it2 != mol2.atom_types_to_indices.end()) ? it2->second : empty_nodes_vec;

        if (nodes1.empty()) {
            for (unsigned int n2_idx : nodes2) total_cost += mol2.atom_data_vec[n2_idx].total_weight;
            continue;
        }
        if (nodes2.empty()) {
            for (unsigned int n1_idx : nodes1) total_cost += mol1.atom_data_vec[n1_idx].total_weight;
            continue;
        }

        size_t n1 = nodes1.size();
        size_t n2 = nodes2.size();
        size_t max_size = std::max(n1, n2);
        std::vector<cost> cost_matrix_flat(max_size * max_size, 0.0);

        // Fill cost matrix
#if FAST_MCES_PROFILE
        auto fill_start = std::chrono::steady_clock::now();
#endif
        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                cost_matrix_flat[i * max_size + j] = node_cost(nodes1[i], mol1, nodes2[j], mol2);
            }
        }
        if (n1 < n2) {
            for (size_t i = n1; i < max_size; ++i) {
                for (size_t j = 0; j < n2; ++j) {
                    cost_matrix_flat[i * max_size + j] = mol2.atom_data_vec[nodes2[j]].total_weight;
                }
            }
        } else if (n2 < n1) {
            for (size_t i = 0; i < n1; ++i) {
                for (size_t j = n2; j < max_size; ++j) {
                    cost_matrix_flat[i * max_size + j] = mol1.atom_data_vec[nodes1[i]].total_weight;
                }
            }
        }
#if FAST_MCES_PROFILE
        auto fill_end = std::chrono::steady_clock::now();
        local_fill_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(fill_end - fill_start).count();
#endif

        // Solve LAP for this block
#if FAST_MCES_PROFILE
        auto solve_start = std::chrono::steady_clock::now();
#endif
        total_cost += solve_lap(cost_matrix_flat, max_size);
#if FAST_MCES_PROFILE
        auto solve_end = std::chrono::steady_clock::now();
        local_solve_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count();
#endif
    }

#if FAST_MCES_PROFILE
    g_pair_fill_time_ns.fetch_add(local_fill_ns);
    g_pair_solve_time_ns.fetch_add(local_solve_ns);
    g_pair_count.fetch_add(1);
#endif

    return total_cost;
}

std::vector<cost> calculate_symmetric_distance_matrix(const std::vector<std::string>& smiles_list) {
    size_t n = smiles_list.size();
    std::vector<PrecomputedMol> precomputed_mols(n);

#if FAST_MCES_PROFILE
    auto t0 = std::chrono::steady_clock::now();
#endif
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        precomputed_mols[i] = precompute_mol_data(smiles_list[i]);
    }
#if FAST_MCES_PROFILE
    auto t1 = std::chrono::steady_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_precompute_time_ns.fetch_add(ns);
    g_precompute_count.fetch_add(static_cast<long long>(n));
#endif

    return filter2_batch_symmetric(precomputed_mols);
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
            if (std::isfinite(dist) && dist >= 0 && dist <= 10000.0) {
                results[i * n + j] = dist;
                results[j * n + i] = dist;
            } else {
                results[i * n + j] = 0.0;
                results[j * n + i] = 0.0;
#if FAST_MCES_ERROR_COUNT
                error_count++;
#endif
            }
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
            cost dist = calculate_pair_distance(mols1[i], mols2[j]);
            if (std::isfinite(dist) && dist >= 0 && dist <= 10000.0) {
                results[i * n2 + j] = dist;
            } else {
                results[i * n2 + j] = 0.0;
#if FAST_MCES_ERROR_COUNT
                error_count++;
#endif
            }
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