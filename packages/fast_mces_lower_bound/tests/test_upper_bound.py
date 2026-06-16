"""Validate the C++ clique-based MCES upper bound against exact count-based MCES."""

import fast_mces_lower_bound
import networkx as nx
from rdkit import Chem


def _atom_label(atom):
    return (atom.GetAtomicNum(), atom.GetDegree(), atom.GetIsAromatic(), atom.GetFormalCharge())


def _bond_label(bond):
    return (bond.GetBondTypeAsDouble(), bond.GetIsAromatic())


def _line_vertex_label(bond):
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    labels = tuple(sorted([_atom_label(a1), _atom_label(a2)]))
    return (_bond_label(bond), labels)


def _shared_atom_label(mol, b1, b2):
    idx1 = {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()}
    idx2 = {b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}
    common = idx1 & idx2
    if not common:
        return None
    atom = mol.GetAtomWithIdx(common.pop())
    return _atom_label(atom)


def _build_association_graph(mol1, mol2):
    bonds1 = list(mol1.GetBonds())
    bonds2 = list(mol2.GetBonds())

    # Compatible line-vertex pairs.
    nodes = []
    for b1 in bonds1:
        lab1 = _line_vertex_label(b1)
        for b2 in bonds2:
            if lab1 == _line_vertex_label(b2):
                nodes.append((b1.GetIdx(), b2.GetIdx()))

    G = nx.Graph()
    G.add_nodes_from(nodes)

    for i, (e1a, e2a) in enumerate(nodes):
        b1a = mol1.GetBondWithIdx(e1a)
        b2a = mol2.GetBondWithIdx(e2a)
        for j in range(i + 1, len(nodes)):
            e1b, e2b = nodes[j]
            if e1a == e1b or e2a == e2b:
                continue
            b1b = mol1.GetBondWithIdx(e1b)
            b2b = mol2.GetBondWithIdx(e2b)

            adj1 = _shared_atom_label(mol1, b1a, b1b) is not None
            adj2 = _shared_atom_label(mol2, b2a, b2b) is not None
            if adj1 != adj2:
                continue
            if adj1:
                if _shared_atom_label(mol1, b1a, b1b) != _shared_atom_label(mol2, b2a, b2b):
                    continue
            G.add_edge((e1a, e2a), (e1b, e2b))
    return G


def _matched_bonds_connected(mol, bond_indices):
    if len(bond_indices) <= 1:
        return True
    adj = {b: set() for b in bond_indices}
    for i, b1 in enumerate(bond_indices):
        for b2 in bond_indices[i + 1:]:
            a1 = {mol.GetBondWithIdx(b1).GetBeginAtomIdx(), mol.GetBondWithIdx(b1).GetEndAtomIdx()}
            a2 = {mol.GetBondWithIdx(b2).GetBeginAtomIdx(), mol.GetBondWithIdx(b2).GetEndAtomIdx()}
            if a1 & a2:
                adj[b1].add(b2)
                adj[b2].add(b1)
    seen = set()
    stack = [bond_indices[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nb in adj[cur]:
            if nb not in seen:
                stack.append(nb)
    return len(seen) == len(bond_indices)


def _exact_mces_size(mol1, mol2, connected=False):
    G = _build_association_graph(mol1, mol2)
    if G.number_of_nodes() == 0:
        exact_size = 0
    else:
        if connected:
            exact_size = 0
            for clique in nx.find_cliques(G):
                bonds1 = [p[0] for p in clique]
                bonds2 = [p[1] for p in clique]
                if _matched_bonds_connected(mol1, bonds1) and _matched_bonds_connected(mol2, bonds2):
                    exact_size = max(exact_size, len(clique))
        else:
            exact_size = max((len(c) for c in nx.find_cliques(G)), default=0)
    return exact_size


def _validate_result(result, mol1, mol2, connected):
    keys = {
        "matched_edge_pairs",
        "matched_edge_count",
        "distance_upper_bound",
        "association_vertex_count",
        "association_edge_count",
        "runtime_ms",
        "metadata",
    }
    assert set(result.keys()) == keys, f"Unexpected keys: {set(result.keys())}"
    meta = result["metadata"]
    assert meta["connected_mode"] == connected
    assert meta["compatibility_mode"] == "exact_bond_atom_labels"
    assert meta["clique_heuristic"] == "multi_start_greedy_degree"
    assert "weighted" not in meta, "Legacy 'weighted' metadata should be removed"

    pairs = result["matched_edge_pairs"]
    assert isinstance(pairs, list)
    assert len(pairs) == result["matched_edge_count"]

    # The C++ upper bound always reports the weighted ILP distance for the
    # clique it finds:
    #   distance = sum of bond weights in G1 + sum in G2
    #              - 2 * sum(min(g1.bond_type, g2.bond_type)) over matched pairs
    def _bond_weight(mol, idx):
        return mol.GetBondWithIdx(idx).GetBondTypeAsDouble()

    total_w1 = sum(b.GetBondTypeAsDouble() for b in mol1.GetBonds())
    total_w2 = sum(b.GetBondTypeAsDouble() for b in mol2.GetBonds())
    matched_min = sum(
        min(_bond_weight(mol1, e1), _bond_weight(mol2, e2)) for e1, e2 in pairs
    )
    expected = total_w1 + total_w2 - 2.0 * matched_min
    assert result["distance_upper_bound"] == expected, (
        f"Weighted distance mismatch: got {result['distance_upper_bound']}, "
        f"expected {expected}"
    )

    seen1, seen2 = set(), set()
    for e1, e2 in pairs:
        assert e1 not in seen1, "Duplicate bond from molecule 1"
        assert e2 not in seen2, "Duplicate bond from molecule 2"
        seen1.add(e1)
        seen2.add(e2)

        b1 = mol1.GetBondWithIdx(e1)
        b2 = mol2.GetBondWithIdx(e2)
        assert _line_vertex_label(b1) == _line_vertex_label(b2), "Vertex compatibility failed"

    # Association-edge consistency for every pair of matched pairs.
    for i, (e1a, e2a) in enumerate(pairs):
        b1a = mol1.GetBondWithIdx(e1a)
        b2a = mol2.GetBondWithIdx(e2a)
        for e1b, e2b in pairs[i + 1:]:
            b1b = mol1.GetBondWithIdx(e1b)
            b2b = mol2.GetBondWithIdx(e2b)
            adj1 = _shared_atom_label(mol1, b1a, b1b) is not None
            adj2 = _shared_atom_label(mol2, b2a, b2b) is not None
            assert adj1 == adj2, "Adjacency consistency violated"
            if adj1:
                assert _shared_atom_label(mol1, b1a, b1b) == _shared_atom_label(mol2, b2a, b2b), "Shared atom label mismatch"

    if connected:
        bonds1 = [p[0] for p in pairs]
        bonds2 = [p[1] for p in pairs]
        assert _matched_bonds_connected(mol1, bonds1), "Connected mode violated in molecule 1"
        assert _matched_bonds_connected(mol2, bonds2), "Connected mode violated in molecule 2"


def _test_pair(s1, s2, connected=False):
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)
    assert mol1 is not None and mol2 is not None

    result = fast_mces_lower_bound.mces_distance_upper_bound(
        s1, s2, {"connected": connected, "num_starts": 200},
    )
    _validate_result(result, mol1, mol2, connected)

    exact_size = _exact_mces_size(mol1, mol2, connected=connected)
    # The exact reference we have here is in the unweighted count metric, so
    # it isn't directly comparable to the weighted distance. We instead
    # validate (a) the clique size does not exceed the exact maximum clique
    # size, and (b) the weighted distance is non-negative and matches the
    # weighted formula exactly (validated by _validate_result).
    assert result["matched_edge_count"] <= exact_size, (
        f"Upper bound exceeded exact MCES size for {s1} vs {s2} (connected={connected}): "
        f"{result['matched_edge_count']} > {exact_size}"
    )
    assert result["distance_upper_bound"] >= 0.0, (
        f"Negative weighted distance for {s1} vs {s2}: "
        f"{result['distance_upper_bound']}"
    )

    print(f"  {s1:25} vs {s2:25}  connected={connected}  "
          f"upper_k={result['matched_edge_count']}  exact_k={exact_size}  "
          f"upper_d={result['distance_upper_bound']}")


def main():
    pairs = [
        ("CCO", "CCO"),
        ("CC=O", "CC=O"),
        ("CC", "CC"),
        ("CC", "CCC"),
        ("CCO", "CC=O"),
        ("CCN", "CCO"),
        ("CCC", "CCO"),
        ("c1ccccc1", "c1ccccc1"),
        ("c1ccccc1", "c1ccccc1C"),
        ("Nc1cc(C)ccc1", "Nc1ccc(C)cc1"),
        ("C1CCCCC1", "C1CCCCC1C"),
        ("CC(C)C", "CC(C)N"),
        ("[CH3-]", "C"),
        ("C[C@H](O)N", "C[C@@H](O)N"),
    ]

    print("Disconnected mode:")
    for s1, s2 in pairs:
        _test_pair(s1, s2, connected=False)

    print("\nConnected mode subset:")
    connected_pairs = [
        ("CCO", "CCO"),
        ("CC", "CCC"),
        ("c1ccccc1", "c1ccccc1C"),
        ("C1CCCCC1", "C1CCCCC1C"),
        ("CC.CC", "CCCC"),
        ("CCO", "CC=O"),
    ]
    for s1, s2 in connected_pairs:
        _test_pair(s1, s2, connected=True)

    print("\nAll upper-bound tests passed!")


if __name__ == "__main__":
    main()
