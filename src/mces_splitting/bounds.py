from typing import List, Tuple, Optional, Generator, Iterable , Sequence
from scipy.optimize import linear_sum_assignment
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
from fast_mces_lower_bound import calculate_symmetric_distance_matrix, calculate_distance_matrix

def filter1(G1: nx.Graph, G2: nx.Graph) -> float:
    """
     Finds a lower bound for the distance based on degree

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.

     Returns:
     -------
     float
         Lower bound for the distance between the molecules

    """
    #Find all occuring atom types and partition by type
    atom_types1=[]
    for i in G1.nodes:
        if G1.nodes[i]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[i]["atom"])
    type_map1={}
    for i in atom_types1:
        type_map1[i]=list(filter(lambda x: i==G1.nodes[x]["atom"],G1.nodes))

    atom_types2=[]
    for i in G2.nodes:
        if G2.nodes[i]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[i]["atom"])
    type_map2={}
    for i in atom_types2:
        type_map2[i]=list(filter(lambda x: i==G2.nodes[x]["atom"],G2.nodes))

    #calculate lower bound
    difference=0
    #Every atom type is done seperately
    for i in atom_types1:
        if i in atom_types2:
            #number of nodes that can be mapped
            n=min(len(type_map1[i]),len(type_map2[i]))
            #sort by degree
            degreelist1=sorted(type_map1[i],key=lambda x:sum([G1[x][j]["weight"] for j in G1.neighbors(x)]),reverse=True)
            degreelist2=sorted(type_map2[i],key=lambda x:sum([G2[x][j]["weight"] for j in G2.neighbors(x)]),reverse=True)
            #map in order of sorted lists
            for j in range(n):
                deg1=sum([G1[degreelist1[j]][k]["weight"] for k in G1.neighbors(degreelist1[j])])
                deg2=sum([G2[degreelist2[j]][k]["weight"] for k in G2.neighbors(degreelist2[j])])
                difference+= abs(deg1-deg2)
            #nodes that are not mapped
            if len(degreelist1)>n:
                for j in range(n,len(degreelist1)):
                    difference+=sum([G1[degreelist1[j]][k]["weight"] for k in G1.neighbors(degreelist1[j])])
            if len(degreelist2)>n:
                for j in range(n,len(degreelist2)):
                    difference+=sum([G2[degreelist2[j]][k]["weight"] for k in G2.neighbors(degreelist2[j])])
        #atom type only in one of the graphs
        else:
            for j in type_map1[i]:
                difference+=sum([G1[j][k]["weight"] for k in G1.neighbors(j)])
    for i in atom_types2:
        if i not in atom_types1:
            for j in type_map2[i]:
                difference+=sum([G2[j][k]["weight"] for k in G2.neighbors(j)])
    return difference/2

def _get_cost(G1: nx.Graph, G2: nx.Graph, i: int, j: int) -> float:
    """
     Calculates the cost for mapping node i to j based on neighborhood

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.
     i : int
         Node of G1
     j : int
         Node of G2

     Returns:
     -------
     float
         Cost of mapping i to j

    """
    #Find all occuring atom types in neighborhood
    atom_types1=[]
    for k in G1.neighbors(i):
        if G1.nodes[k]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[k]["atom"])
    type_map1={}
    for k in atom_types1:
        type_map1[k]=list(filter(lambda x: k==G1.nodes[x]["atom"],G1.neighbors(i)))


    atom_types2=[]
    for k in G2.neighbors(j):
        if G2.nodes[k]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[k]["atom"])
    type_map2={}
    for k in atom_types2:
        type_map2[k]=list(filter(lambda x: k==G2.nodes[x]["atom"],G2.neighbors(j)))

    #calculate cost
    difference=0.
    #Every atom type is handled seperately
    for k in atom_types1:
        if k in atom_types2:
            n=min(len(type_map1[k]),len(type_map2[k]))
            #sort by incident edges by weight
            edgelist1=sorted(type_map1[k],key=lambda x:G1[i][x]["weight"],reverse=True)
            edgelist2=sorted(type_map2[k],key=lambda x:G2[j][x]["weight"],reverse=True)
            #map in order of sorted lists
            for l in range(n):
                difference+=(max(G1[i][edgelist1[l]]["weight"],G2[j][edgelist2[l]]["weight"])-min(G1[i][edgelist1[l]]["weight"],G2[j][edgelist2[l]]["weight"]))/2
            #cost for not mapped edges
            if len(edgelist1)>n:
                for l in range(n,len(edgelist1)):
                    difference+=G1[i][edgelist1[l]]["weight"]/2
            if len(edgelist2)>n:
                for l in range(n,len(edgelist2)):
                    difference+=G2[j][edgelist2[l]]["weight"]/2
        else:
            for l in type_map1[k]:
                difference+=G1[i][l]["weight"]/2
    for k in atom_types2:
        if k not in atom_types1:
            for l in type_map2[k]:
                difference+=G2[j][l]["weight"]/2
    return difference

def filter2_from_lib(G1: nx.Graph, G2: nx.Graph):
    """
     Finds a lower bound for the distance based on neighborhood

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.

     Returns:
     -------
     float
         Lower bound for the distance between the molecules

    """
    # Find all occuring atom types
    atom_types1=[]
    for i in G1.nodes:
        if G1.nodes[i]["atom"] not in atom_types1:
            atom_types1.append(G1.nodes[i]["atom"])

    atom_types2=[]
    for i in G2.nodes:
        if G2.nodes[i]["atom"] not in atom_types2:
            atom_types2.append(G2.nodes[i]["atom"])

    atom_types=atom_types1

    for i in atom_types2:
        if i not in atom_types:
            atom_types.append(i)
    #calculate distance
    res=0
    #handle every atom type seperately
    for i in atom_types:
        #filter by atom type
        nodes1=list(filter(lambda x: i==G1.nodes[x]["atom"],G1.nodes))
        nodes2=list(filter(lambda x: i==G2.nodes[x]["atom"],G2.nodes))
        #Create new graph for and solve minimum weight full matching
        G=nx.Graph()
        #Add node for every node of type i in G1 and G2
        for j in nodes1:
            G.add_node(tuple([1,j]))
        for j in nodes2:
            G.add_node(tuple([2,j]))
        #Add edges between all nodes of G1 and G2
        for j in nodes1:
            for k in nodes2:
                if G1.nodes[j]["atom"]==G2.nodes[k]["atom"]:
                    G.add_edge(tuple([1,j]),tuple([2,k]),weight=_get_cost(G1,G2,j,k))
        #Add nodes if one graph has more nodes of type i than the other
        if len(nodes1)<len(nodes2):
            diff=len(nodes2)-len(nodes1)
            for j in range(1,diff+1):
                G.add_node(tuple([1,-j]))
                for k in nodes2:
                    G.add_edge(tuple([1,-j]),tuple([2,k]),weight=sum([G2[l][k]["weight"] for l in G2.neighbors(k)])/2)
        if len(nodes2)<len(nodes1):
            diff=len(nodes1)-len(nodes2)
            for j in range(1,diff+1):
                G.add_node(tuple([2,-j]))
                for k in nodes1:
                    G.add_edge(tuple([1,k]),tuple([2,-j]),weight=sum([G1[l][k]["weight"] for l in G1.neighbors(k)])/2)
        #Solve minimum weight full matching
        h=nx.bipartite.minimum_weight_full_matching(G)
        #Add weight of the matching
        for k in h:
            if k[0]==1:
                res=res+G[k][h[k]]["weight"]

    return res

def mces_lower_bound_symmetric(smiles_list:Sequence[str]) -> NDArray:
    """
    Wrapper for the fast C++ MCES bounds calculation using SMILES strings directly.
    This uses the optimized C++ implementation with parallel processing.
    
    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings representing molecules
        
    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix where result[i,j] is the distance between molecules i and j
    """
    symmetric_distance_matrix = calculate_symmetric_distance_matrix(smiles_list)
    # we get a flattened list and now format it as a square numpy array
    # symmetric_distance_matrix is expected to be a flat list or 1D numpy array of length n*n
    n = int(np.sqrt(len(symmetric_distance_matrix)))
    return np.array(symmetric_distance_matrix).reshape((n, n))

def mces_lower_bound(smiles_list1: Sequence[str], smiles_list2: Sequence[str]) -> NDArray:
    """
    Wrapper for the fast C++ MCES bounds calculation using SMILES strings directly.
    This uses the optimized C++ implementation with parallel processing.

    Parameters
    ----------
    smiles_list1 : list of str
        List of SMILES strings representing molecules
    smiles_list2 : list of str
        List of SMILES strings representing molecules

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix where result[i,j] is the distance between molecules i and j
    """
    distance_matrix = calculate_distance_matrix(smiles_list1, smiles_list2)
    return distance_matrix

if __name__ == "__main__":
    import sys
    from time import perf_counter
    from typing import List
    import polars as pl
    import os
    from .graph_construction import construct_graph

    if '--bounds-validity-test' in sys.argv:
        skip_mces = '--no-mces' in sys.argv
        from .lib import MCES_ILP
        data_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
        if len(sys.argv) > sys.argv.index('--bounds-validity-test') + 1:
            try:
                number_of_mol:int = int(sys.argv[sys.argv.index('--bounds-validity-test') + 1])
            except Exception:
                try:
                    number_of_mol:int = int(sys.argv[sys.argv.index('--bounds-validity-test') + 2])
                except Exception:
                    number_of_mol:int = 10
        else:
            number_of_mol:int = 10
        print(f"Running bounds validity test on {number_of_mol} molecules, skip_mces={skip_mces}")
        smiles_examples = pl.scan_csv(data_file_path).head(number_of_mol).collect().to_series().to_list()
        graphs = [construct_graph(smiles) for smiles in smiles_examples]


        # Test C++ implementation if available
        try:
            start_time = perf_counter()
            cpp_matrix = mces_lower_bound_symmetric(smiles_examples)
            filter2_cpp_results = cpp_matrix.flatten()
            time2_cpp = perf_counter() - start_time
            cpp_available = True
            print("C++ implementation available and tested")
        except ImportError:
            print("C++ implementation not available, skipping C++ tests")
            cpp_available = False
            filter2_cpp_results = None
            time2_cpp = 0

        # time the filters
        start_time = perf_counter()
        filter1_results = [filter1(G1, G2) for G1 in graphs for G2 in graphs]
        time1 = perf_counter() - start_time

        start_time = perf_counter()
        filter2_from_lib_results = [filter2_from_lib(G1, G2) for G1 in graphs for G2 in graphs]
        time2_from_lib = perf_counter() - start_time

        start_time = perf_counter()
        batch_matrix = filter2_batch(graphs)
        filter2_batch_results = batch_matrix.flatten()
        time2_batch = perf_counter() - start_time

        # Compute true MCES distances (this will be the slowest) - only if not disabled
        if not skip_mces:
            start_time = perf_counter()
            mces_results = []
            for i, G1 in enumerate(graphs):
                for j, G2 in enumerate(graphs):
                    if i == j:
                        mces_results.append(0.0)
                    else:
                        try:
                            distance, distance_type = MCES_ILP(G1, G2,threshold=1000,no_ilp_threshold=True,solver="default")
                            mces_results.append(distance)
                        except Exception as e:
                            print(f"MCES_ILP failed for graphs {i}, {j}: {e}")
                            mces_results.append(float('inf'))
            time_mces = perf_counter() - start_time
        else:
            print("Skipping MCES calculation (--no-mces flag provided)")
            mces_results = [0.0] * len(filter1_results)
            time_mces = 0

        # Check for invalid bounds (any filter result > true MCES distance) - only if MCES was computed
        if not skip_mces:
            invalid_bounds_found = False
            for i, (f1, f2_lib, f2_batch, mces) in enumerate(zip(
                filter1_results, filter2_from_lib_results, filter2_batch_results, mces_results)):
                if mces == float('inf'):
                    continue
                filters = [("Filter1", f1), ("Filter2_lib", f2_lib), ("Filter2_batch", f2_batch)]
                if cpp_available and filter2_cpp_results is not None:
                    filters.append(("Filter2_cpp", filter2_cpp_results[i]))
                for filter_name, filter_result in filters:
                    if filter_result > mces + 1e-6:
                        print(f"INVALID BOUND ALERT! Test {i}: {filter_name} = {filter_result:.6f} > MCES = {mces:.6f}")
                        invalid_bounds_found = True

        # Print results where filter2 variants differ
        for i, (f1, f2_lib, f2_batch) in enumerate(zip(
            filter1_results, filter2_from_lib_results, filter2_batch_results)):
            mces = mces_results[i] if not skip_mces else "N/A"
            comparison_values = [f2_lib, f2_batch]
            if cpp_available and filter2_cpp_results is not None:
                f2_cpp = filter2_cpp_results[i]
                comparison_values.append(f2_cpp)
            all_close = True
            for j in range(len(comparison_values)):
                for k in range(j+1, len(comparison_values)):
                    if not np.isclose(comparison_values[j], comparison_values[k]):
                        all_close = False
                        break
                if not all_close:
                    break
            if not all_close:
                print(f"Test {i}:")
                print(f"  Filter1: {f1:.6f}")
                print(f"  Filter2 from lib: {f2_lib:.6f}")
                print(f"  Filter2 batch: {f2_batch:.6f}")
                if cpp_available and filter2_cpp_results is not None:
                    print(f"  Filter2 C++: {f2_cpp:.6f}")
                print(f"  MCES (true): {mces}")
                print("  ALERT: Filter2 variants differ!")

        # Print summary statistics - only if MCES was computed
        if not skip_mces:
            valid_indices = [i for i, mces in enumerate(mces_results) if mces != float('inf')]
            if valid_indices:
                print(f"\nSummary for {len(valid_indices)} valid MCES computations:")
                all_consistent = all(
                    np.isclose(filter2_from_lib_results[i], filter2_batch_results[i]) and
                    (not cpp_available or filter2_cpp_results is None or np.isclose(filter2_from_lib_results[i], filter2_cpp_results[i]))
                    for i in valid_indices
                )
                if all_consistent:
                    filter_names = "filter2_from_lib and filter2_batch"
                    if cpp_available and filter2_cpp_results is not None:
                        filter_names += ", and filter2_cpp"
                    print(f"All results of {filter_names} are consistent.")
                    avg_diff_f2 = sum((filter2_from_lib_results[i] - filter1_results[i]) for i in valid_indices) / len(valid_indices)
                    avg_gap_f1 = sum((mces_results[i] - filter1_results[i]) for i in valid_indices) / len(valid_indices)
                    avg_gap_f2 = sum((mces_results[i] - filter2_from_lib_results[i]) for i in valid_indices) / len(valid_indices)
                    print("Average improvement over filter1:")
                    print(f"  Filter2 from lib - Filter1: {avg_diff_f2:.4f}")
                    print("Average gap to true MCES distance:")
                    print(f"  MCES - Filter1: {avg_gap_f1:.4f}")
                    print(f"  MCES - Filter2 from lib: {avg_gap_f2:.4f}")
            if not invalid_bounds_found:
                print("\n✓ All bounds are valid (no filter exceeded true MCES distance)")
            else:
                print("\n⚠️  INVALID BOUNDS DETECTED! Check the algorithms above.")
        else:
            all_consistent = all(
                np.isclose(filter2_from_lib_results[i], filter2_batch_results[i]) and
                (not cpp_available or filter2_cpp_results is None or np.isclose(filter2_from_lib_results[i], filter2_cpp_results[i]))
                for i in range(len(filter2_from_lib_results))
            )
            if all_consistent:
                filter_names = "filter2_from_lib and filter2_batch"
                if cpp_available and filter2_cpp_results is not None:
                    filter_names += ", and filter2_cpp"
                print(f"\nAll results of {filter_names} are consistent.")
            else:
                print("\n⚠️  Filter variants produce inconsistent results!")
            avg_diff_f2 = sum((filter2_from_lib_results[i] - filter1_results[i]) for i in range(len(filter1_results))) / len(filter1_results)
            print("\nAverage improvement over filter1 (tightness):")
            print(f"  Filter2 from lib - Filter1: {avg_diff_f2:.4f}")

        print("\nTiming results:")
        print(f"Time for filter1: {time1:.2f} seconds")
        print(f"Time for filter2 from lib: {time2_from_lib:.2f} seconds")
        print(f"Time for filter2_batch: {time2_batch:.2f} seconds")
        if cpp_available:
            print(f"Time for filter2_cpp: {time2_cpp:.2f} seconds")
        if not skip_mces:
            print(f"Time for MCES_ILP (true): {time_mces:.2f} seconds")
        else:
            print("MCES_ILP calculation skipped")

    if '--cpp-test' in sys.argv:
        data_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
        number_of_mol:int = 200
        smiles = pl.scan_csv(data_file_path).head(number_of_mol).collect().to_series().to_list()
        start_time = perf_counter()
        # Only use filter2_batch and filter2_cpp
        graphs = [construct_graph(smiles) for smiles in smiles]
        batch_matrix = filter2_batch(graphs)
        if batch_matrix.shape[0] != batch_matrix.shape[1]:
            raise ValueError("The input graphs must be a square matrix (same number of graphs in both lists).")
        upper_triangle_indices = np.triu_indices(batch_matrix.shape[0], k=1)
        filter2_batch_results = batch_matrix[upper_triangle_indices].flatten()
        time2_batch = perf_counter() - start_time
        print(f"Time for filter2_batch on DSSTox dataset: {time2_batch:.2f} seconds")
        start_time = perf_counter()
        print("Running C++ implementation on SMILES data...")
        cpp_results = mces_lower_bound_symmetric(smiles)
        if cpp_results.shape[0] != cpp_results.shape[1]:
            raise ValueError("The input graphs must be a square matrix (same number of graphs in both lists).")
        upper_triangle_indices = np.triu_indices(cpp_results.shape[0], k=1)
        filter2_cpp_results = cpp_results[upper_triangle_indices].flatten()
        time2_cpp = perf_counter() - start_time
        print(f"Time for filter2_cpp on DSSTox dataset: {time2_cpp:.2f} seconds")
        if len(filter2_batch_results) != len(filter2_cpp_results):
            raise ValueError("The number of results from filter2_batch and filter2_cpp must match.")
        differences = np.abs(filter2_batch_results - filter2_cpp_results)
        max_difference = np.max(differences)
        if max_difference > 1e-6:
            print(f"Results differ! Maximum difference: {max_difference:.6f}")
        else:
            print("Results are consistent between filter2_batch and filter2_cpp.")

    if "--cpp-benchmark" in sys.argv:
        from ..rdkit.mol import sanitize_smiles_polars
        data_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
        number_of_mol:int = 2000
        smiles = pl.scan_csv(data_file_path).with_columns(
            pl.col("MS_READY_SMILES").map_batches(
                function=sanitize_smiles_polars,
                return_dtype=pl.String,
            )
        ).filter(pl.col("MS_READY_SMILES").is_not_null()).head(number_of_mol).collect().to_series().to_list()
        start_time = perf_counter()
        print("Running C++ implementation on SMILES data...")
        cpp_results = mces_lower_bound_symmetric(smiles)
        if cpp_results.shape[0] != cpp_results.shape[1]:
            raise ValueError("The input graphs must be a square matrix (same number of graphs in both lists).")
        upper_triangle_indices = np.triu_indices(cpp_results.shape[0], k=1)
        filter2_cpp_results = cpp_results[upper_triangle_indices].flatten()
        time2_cpp = perf_counter() - start_time
        print(f"Time for filter2_cpp on DSSTox dataset with {number_of_mol} molecules: {time2_cpp:.2f} seconds")
        print(f"Number of comparisons: {len(filter2_cpp_results)}")
        print(f"Average time per comparison: {time2_cpp / len(filter2_cpp_results):.6f} seconds")
        print(f"Total number of molecules: {number_of_mol}")
        print(f"Total number of comparisons: {number_of_mol * (number_of_mol - 1) // 2}")

    if '--bounds-strength-test' in sys.argv:
        from .lib import MCES_ILP
        from .mces import calculate_mces_distances, suppress_output
        data_file_path = os.path.join(os.path.dirname(__file__), "dsstox_smiles_medium.csv")
        number_of_mol:int = 10
        smiles = pl.scan_csv(data_file_path).head(number_of_mol).collect().to_series().to_list()
        graphs = [construct_graph(smiles) for smiles in smiles]
        start_time = perf_counter()
        batch_matrix = filter2_batch(graphs)
        if batch_matrix.shape[0] != batch_matrix.shape[1]:
            raise ValueError("The input graphs must be a square matrix (same number of graphs in both lists).")
        upper_triangle_indices = np.triu_indices(batch_matrix.shape[0], k=1)
        filter2_batch_results = batch_matrix[upper_triangle_indices].flatten()
        time2_batch = perf_counter() - start_time
        print(f"Time for filter2_batch on DSSTox dataset: {time2_batch:.2f} seconds")
        start_time = perf_counter()
        mces_results = []
        with suppress_output():
            for i, G1 in enumerate(graphs):
                for j, G2 in enumerate(graphs):
                    if i == j:
                        continue
                    elif i < j:
                        try:
                            distance, distance_type = MCES_ILP(G1, G2, threshold=10, no_ilp_threshold=True, solver="gurobi")
                            mces_results.append(distance)
                        except Exception as e:
                            print(f"MCES_ILP failed for graphs {i}, {j}: {e}")
                            mces_results.append(float('inf'))
                    elif i > j:
                        continue
        time_mces = perf_counter() - start_time
        print(f"Time for filter2_batch on DSSTox dataset: {time2_batch:.2f} seconds")
        print(f"Time for MCES_ILP (true) on DSSTox dataset: {time_mces:.2f} seconds")
        count_mces_greater = 0
        diffs = []
        for mces, bound in zip(mces_results, filter2_batch_results):
            if mces > bound:
                count_mces_greater += 1
                diffs.append(mces - bound)
        print(f"Number of comparisons where MCES > filter2_batch bound: {count_mces_greater} out of {len(mces_results)} total comparisons")
        if count_mces_greater > 0:
            avg_diff = sum(diffs) / count_mces_greater
            print(f"Average difference (MCES - bound) for those cases: {avg_diff:.4f}")
        else:
            print("No cases where MCES > filter2_batch bound.")
        print(f"speedup in time: {time_mces / time2_batch:.2f}x")