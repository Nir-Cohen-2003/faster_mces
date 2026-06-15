#include "mces_upper_bound.hpp"

#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AtomLabel {
    int atomic_num = 0;
    int degree = 0;
    bool aromatic = false;
    int formal_charge = 0;

    bool operator==(const AtomLabel& other) const noexcept {
        return atomic_num == other.atomic_num &&
               degree == other.degree &&
               aromatic == other.aromatic &&
               formal_charge == other.formal_charge;
    }

    bool operator<(const AtomLabel& other) const noexcept {
        if (atomic_num != other.atomic_num) return atomic_num < other.atomic_num;
        if (degree != other.degree) return degree < other.degree;
        if (aromatic != other.aromatic) return aromatic < other.aromatic;
        return formal_charge < other.formal_charge;
    }
};

struct BondInfo {
    int idx = -1;           // RDKit bond index
    int u = -1;             // endpoint atom indices
    int v = -1;
    double bond_type = 0.0; // bond type as double
    bool aromatic = false;
    AtomLabel label_u;
    AtomLabel label_v;
};

struct MoleculeGraph {
    std::unique_ptr<RDKit::ROMol> mol;
    std::vector<BondInfo> bonds;
    int num_bonds = 0;
    // bond_adj[i][j] == 1 iff bonds i and j share an atom in the original molecule
    std::vector<std::vector<char>> bond_adj;
    // shared_label[i][j] valid when bond_adj[i][j] == 1
    std::vector<std::vector<AtomLabel>> shared_label;
};

static AtomLabel make_atom_label(const RDKit::Atom* atom) {
    AtomLabel lab;
    lab.atomic_num = atom->getAtomicNum();
    lab.degree = static_cast<int>(atom->getDegree());
    lab.aromatic = atom->getIsAromatic();
    lab.formal_charge = atom->getFormalCharge();
    return lab;
}

static std::string atom_label_string(const AtomLabel& lab) {
    return std::to_string(lab.atomic_num) + "#" +
           std::to_string(lab.degree) + "#" +
           (lab.aromatic ? "A" : "N") + "#" +
           std::to_string(lab.formal_charge);
}

static std::string bond_label_string(double bond_type, bool aromatic) {
    // Use a normalized representation.
    return std::to_string(bond_type) + "#" + (aromatic ? "A" : "N");
}

static std::string line_vertex_label(const BondInfo& b) {
    AtomLabel a = b.label_u;
    AtomLabel c = b.label_v;
    if (c < a) std::swap(a, c);
    return bond_label_string(b.bond_type, b.aromatic) + "|" +
           atom_label_string(a) + "|" + atom_label_string(c);
}

static MoleculeGraph build_molecule_graph(const std::string& smiles) {
    MoleculeGraph g;
    g.mol.reset(RDKit::SmilesToMol(smiles));
    if (!g.mol) {
        throw std::runtime_error("Failed to parse SMILES: " + smiles);
    }

    const int nb = static_cast<int>(g.mol->getNumBonds());
    g.num_bonds = nb;
    g.bonds.reserve(nb);
    for (const auto* bond : g.mol->bonds()) {
        BondInfo info;
        info.idx = static_cast<int>(bond->getIdx());
        const RDKit::Atom* a1 = bond->getBeginAtom();
        const RDKit::Atom* a2 = bond->getEndAtom();
        info.u = static_cast<int>(a1->getIdx());
        info.v = static_cast<int>(a2->getIdx());
        info.bond_type = bond->getBondTypeAsDouble();
        info.aromatic = bond->getIsAromatic();
        info.label_u = make_atom_label(a1);
        info.label_v = make_atom_label(a2);
        g.bonds.push_back(info);
    }

    g.bond_adj.assign(nb, std::vector<char>(nb, 0));
    g.shared_label.assign(nb, std::vector<AtomLabel>(nb));

    // For each atom, connect all incident bonds in the line graph.
    for (const auto* atom : g.mol->atoms()) {
        std::vector<int> incident;
        for (const auto* nbond : g.mol->atomNeighbors(atom)) {
            const RDKit::Bond* bond = g.mol->getBondBetweenAtoms(
                atom->getIdx(), nbond->getIdx());
            if (bond) incident.push_back(static_cast<int>(bond->getIdx()));
        }
        AtomLabel shared = make_atom_label(atom);
        for (size_t i = 0; i < incident.size(); ++i) {
            for (size_t j = i + 1; j < incident.size(); ++j) {
                int b1 = incident[i];
                int b2 = incident[j];
                g.bond_adj[b1][b2] = 1;
                g.bond_adj[b2][b1] = 1;
                g.shared_label[b1][b2] = shared;
                g.shared_label[b2][b1] = shared;
            }
        }
    }

    return g;
}

struct AssociationVertex {
    int a; // bond index in G1
    int b; // bond index in G2
};

struct AssociationGraph {
    std::vector<AssociationVertex> vertices;
    std::vector<std::vector<int>> neighbors;
    std::vector<int> degree;
};

static AssociationGraph build_association_graph(const MoleculeGraph& g1,
                                                 const MoleculeGraph& g2) {
    AssociationGraph ag;

    // Bucket line-graph vertices by their canonical label.
    std::map<std::string, std::vector<int>> classes1, classes2;
    for (int i = 0; i < g1.num_bonds; ++i) {
        classes1[line_vertex_label(g1.bonds[i])].push_back(i);
    }
    for (int j = 0; j < g2.num_bonds; ++j) {
        classes2[line_vertex_label(g2.bonds[j])].push_back(j);
    }

    for (const auto& [label, vec1] : classes1) {
        auto it = classes2.find(label);
        if (it == classes2.end()) continue;
        const auto& vec2 = it->second;
        for (int a : vec1) {
            for (int b : vec2) {
                ag.vertices.push_back({a, b});
            }
        }
    }

    const int n = static_cast<int>(ag.vertices.size());
    ag.neighbors.assign(n, {});

    for (int p = 0; p < n; ++p) {
        for (int q = p + 1; q < n; ++q) {
            const int a = ag.vertices[p].a;
            const int ap = ag.vertices[q].a;
            const int b = ag.vertices[p].b;
            const int bp = ag.vertices[q].b;
            if (a == ap || b == bp) continue;

            const bool adj1 = g1.bond_adj[a][ap];
            const bool adj2 = g2.bond_adj[b][bp];
            if (adj1 != adj2) continue;
            if (adj1) {
                if (!(g1.shared_label[a][ap] == g2.shared_label[b][bp])) continue;
            }
            ag.neighbors[p].push_back(q);
            ag.neighbors[q].push_back(p);
        }
    }

    ag.degree.resize(n);
    for (int i = 0; i < n; ++i) ag.degree[i] = static_cast<int>(ag.neighbors[i].size());

    return ag;
}

static std::vector<int> intersect_sorted(const std::vector<int>& a,
                                          const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(std::min(a.size(), b.size()));
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            out.push_back(a[i]);
            ++i; ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return out;
}

static bool is_adjacent_to_any(int bond_idx,
                               const std::vector<int>& selected_bonds,
                               const std::vector<std::vector<char>>& bond_adj) {
    for (int sb : selected_bonds) {
        if (bond_adj[bond_idx][sb]) return true;
    }
    return false;
}

static int select_best_candidate(const std::vector<int>& candidates,
                                 const std::vector<std::vector<int>>& neighbors,
                                 std::vector<char>& scratch) {
    // Choose the candidate with the maximum number of neighbors inside candidates.
    int best = candidates[0];
    int best_score = -1;
    std::fill(scratch.begin(), scratch.end(), 0);
    for (int v : candidates) scratch[v] = 1;
    for (int v : candidates) {
        int score = 0;
        for (int nb : neighbors[v]) {
            if (scratch[nb]) ++score;
        }
        if (score > best_score) {
            best_score = score;
            best = v;
        }
    }
    for (int v : candidates) scratch[v] = 0;
    return best;
}

static std::vector<int> greedy_clique(const AssociationGraph& ag,
                                      const MoleculeGraph& g1,
                                      const MoleculeGraph& g2,
                                      bool connected,
                                      int num_starts) {
    const int n = static_cast<int>(ag.vertices.size());
    if (n == 0) return {};

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int x, int y) { return ag.degree[x] > ag.degree[y]; });

    const int n_starts = std::min(n, std::max(1, num_starts));

    std::vector<char> scratch(n, 0);
    std::vector<int> best_clique;

    for (int s_idx = 0; s_idx < n_starts; ++s_idx) {
        int start = order[s_idx];
        std::vector<int> clique = {start};
        std::vector<int> candidates = ag.neighbors[start];
        // Keep candidates sorted for deterministic intersection.
        std::sort(candidates.begin(), candidates.end());

        std::vector<int> selected_g1_bonds = {ag.vertices[start].a};
        std::vector<int> selected_g2_bonds = {ag.vertices[start].b};

        while (!candidates.empty()) {
            if (connected && static_cast<int>(clique.size()) >= 1) {
                std::vector<int> filtered;
                filtered.reserve(candidates.size());
                for (int v : candidates) {
                    if (is_adjacent_to_any(ag.vertices[v].a, selected_g1_bonds, g1.bond_adj) &&
                        is_adjacent_to_any(ag.vertices[v].b, selected_g2_bonds, g2.bond_adj)) {
                        filtered.push_back(v);
                    }
                }
                if (filtered.empty()) break;
                candidates.swap(filtered);
            }

            int v = select_best_candidate(candidates, ag.neighbors, scratch);
            clique.push_back(v);
            selected_g1_bonds.push_back(ag.vertices[v].a);
            selected_g2_bonds.push_back(ag.vertices[v].b);

            std::vector<int> v_neighbors = ag.neighbors[v];
            std::sort(v_neighbors.begin(), v_neighbors.end());
            candidates = intersect_sorted(candidates, v_neighbors);
        }

        // Simple local improvement: extend the clique while possible.
        while (true) {
            std::vector<char> in_clique(n, 0);
            for (int v : clique) in_clique[v] = 1;

            std::vector<int> common;
            bool first = true;
            for (int v : clique) {
                if (first) {
                    common = ag.neighbors[v];
                    first = false;
                } else {
                    common = intersect_sorted(common, ag.neighbors[v]);
                }
                if (common.empty()) break;
            }

            int add = -1;
            for (int v : common) {
                if (!in_clique[v]) {
                    if (!connected ||
                        (is_adjacent_to_any(ag.vertices[v].a, selected_g1_bonds, g1.bond_adj) &&
                         is_adjacent_to_any(ag.vertices[v].b, selected_g2_bonds, g2.bond_adj))) {
                        add = v;
                        break;
                    }
                }
            }
            if (add == -1) break;

            clique.push_back(add);
            selected_g1_bonds.push_back(ag.vertices[add].a);
            selected_g2_bonds.push_back(ag.vertices[add].b);
        }

        if (clique.size() > best_clique.size()) {
            best_clique = std::move(clique);
        }
    }

    return best_clique;
}

static bool matched_set_connected(const std::vector<int>& bond_indices,
                                  const MoleculeGraph& g) {
    if (bond_indices.size() <= 1) return true;
    std::vector<char> visited(g.num_bonds, 0);
    std::vector<int> stack;
    stack.push_back(bond_indices[0]);
    visited[bond_indices[0]] = 1;
    size_t reached = 0;
    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        ++reached;
        for (int other : bond_indices) {
            if (!visited[other] && g.bond_adj[cur][other]) {
                visited[other] = 1;
                stack.push_back(other);
            }
        }
    }
    return reached == bond_indices.size();
}

static void validate_result(const McesUpperBoundResult& res,
                            const AssociationGraph& ag,
                            const MoleculeGraph& g1,
                            const MoleculeGraph& g2) {
    const int k = res.matched_edge_count;
    if (static_cast<int>(res.matched_edge_pairs.size()) != k) {
        throw std::runtime_error("matched_edge_count does not match number of pairs");
    }

    std::vector<int> bonds1, bonds2;
    bonds1.reserve(k);
    bonds2.reserve(k);

    for (int i = 0; i < k; ++i) {
        int e1 = res.matched_edge_pairs[i].first;
        int e2 = res.matched_edge_pairs[i].second;
        if (e1 < 0 || e1 >= g1.num_bonds || e2 < 0 || e2 >= g2.num_bonds) {
            throw std::runtime_error("Matched bond index out of range");
        }
        if (line_vertex_label(g1.bonds[e1]) != line_vertex_label(g2.bonds[e2])) {
            throw std::runtime_error("Vertex compatibility failed");
        }
        bonds1.push_back(e1);
        bonds2.push_back(e2);
    }

    std::sort(bonds1.begin(), bonds1.end());
    std::sort(bonds2.begin(), bonds2.end());
    if (std::unique(bonds1.begin(), bonds1.end()) != bonds1.end() ||
        std::unique(bonds2.begin(), bonds2.end()) != bonds2.end()) {
        throw std::runtime_error("Duplicate bond in matched set");
    }

    // Build reverse map from association vertex to its index for clique validation.
    std::map<std::pair<int, int>, int> pair_to_assoc;
    for (int i = 0; i < static_cast<int>(ag.vertices.size()); ++i) {
        pair_to_assoc[{ag.vertices[i].a, ag.vertices[i].b}] = i;
    }

    for (int i = 0; i < k; ++i) {
        auto it = pair_to_assoc.find(res.matched_edge_pairs[i]);
        if (it == pair_to_assoc.end()) {
            throw std::runtime_error("Matched pair not present in association graph");
        }
        int vi = it->second;
        for (int j = i + 1; j < k; ++j) {
            auto it2 = pair_to_assoc.find(res.matched_edge_pairs[j]);
            if (it2 == pair_to_assoc.end()) {
                throw std::runtime_error("Matched pair not present in association graph");
            }
            int vj = it2->second;
            const int a1 = ag.vertices[vi].a;
            const int a2 = ag.vertices[vj].a;
            const int b1 = ag.vertices[vi].b;
            const int b2 = ag.vertices[vj].b;
            if (a1 == a2 || b1 == b2) {
                throw std::runtime_error("Association edge uniqueness violated");
            }
            const bool adj1 = g1.bond_adj[a1][a2];
            const bool adj2 = g2.bond_adj[b1][b2];
            if (adj1 != adj2) {
                throw std::runtime_error("Association edge adjacency consistency violated");
            }
            if (adj1 && !(g1.shared_label[a1][a2] == g2.shared_label[b1][b2])) {
                throw std::runtime_error("Association edge label consistency violated");
            }
        }
    }

    if (res.connected_mode) {
        if (!matched_set_connected(bonds1, g1) || !matched_set_connected(bonds2, g2)) {
            throw std::runtime_error("Connected mode violated");
        }
    }
}

} // anonymous namespace

McesUpperBoundResult mces_distance_upper_bound(const std::string& smiles1,
                                                const std::string& smiles2,
                                                bool connected,
                                                int num_starts) {
    using namespace std::chrono;
    auto t0 = steady_clock::now();

    MoleculeGraph g1 = build_molecule_graph(smiles1);
    MoleculeGraph g2 = build_molecule_graph(smiles2);
    AssociationGraph ag = build_association_graph(g1, g2);

    std::vector<int> clique = greedy_clique(ag, g1, g2, connected, num_starts);

    McesUpperBoundResult res;
    res.connected_mode = connected;
    res.compatibility_mode = "exact_bond_atom_labels";
    res.clique_heuristic = "multi_start_greedy_degree";
    res.random_seed = std::nullopt;

    res.matched_edge_count = static_cast<int>(clique.size());
    res.matched_edge_pairs.reserve(clique.size());
    for (int v : clique) {
        res.matched_edge_pairs.emplace_back(ag.vertices[v].a, ag.vertices[v].b);
    }

    res.association_vertex_count = static_cast<int>(ag.vertices.size());
    int edge_count = 0;
    for (const auto& nbrs : ag.neighbors) edge_count += static_cast<int>(nbrs.size());
    res.association_edge_count = edge_count / 2;

    res.distance_upper_bound = g1.num_bonds + g2.num_bonds - 2 * res.matched_edge_count;

    auto t1 = steady_clock::now();
    res.runtime_ms = static_cast<int>(duration_cast<milliseconds>(t1 - t0).count());

    validate_result(res, ag, g1, g2);
    return res;
}
