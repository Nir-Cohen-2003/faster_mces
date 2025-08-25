import numpy as np
from numpy.typing import NDArray
import polars as pl
import os
from multiprocessing import cpu_count
from typing import List, Optional, Generator, Iterable, Tuple, Dict
from itertools import batched, chain
from contextlib import contextmanager
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from .bounds import mces_lower_bound, mces_lower_bound_symmetric
from functools import lru_cache
import pulp
import networkx as nx
from rdkit import Chem



if __name__ == "__main__":
    PROFILE = True
else:
    PROFILE = False
    
def calculate_mces_distances(
        smiles_list1: List[str], smiles_list2: Optional[List[str]] = None,
        n_jobs: int = -1, batch_size: int = 20, threshold: int = 10, solver: str = "GUROBI") -> NDArray[np.int64]:
    """
    Efficiently computes exact MCES distances between all pairs of molecules.
    
    Parameters
    ----------
    smiles_list1 : List[str]
        List of SMILES strings for the first set of molecules
    smiles_list2 : Optional[List[str]]
        List of SMILES strings for the second set of molecules.
        If None and symmetric=True, will compare molecules within smiles_list1.
    n_jobs : int
        Number of parallel jobs to run. -1 means use all available cores.
    symmetric : bool
        If True, optimize for comparing molecules within the same list.
    batch_size : int
        Number of pairs to process in each parallel job to reduce overhead.
        
    Returns
    -------
    np.ndarray
        Matrix where element [i,j] is the exact MCES distance between molecules i and j
    """
    assert threshold > 0
    assert smiles_list1 is not None and len(smiles_list1) > 0, "smiles_list1 is empty"
    if smiles_list2 is None:
        symmetric = True
    else:
        symmetric = False
        assert len(smiles_list2) > 0, "smiles_list2 is empty"


    if PROFILE:
        total_start = time.perf_counter()
        print("\n=== Profiling calculate_mces_distances ===")
    
    if n_jobs == -1:
        n_jobs: int = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1
    elif smiles_list2 is None:
        raise ValueError("smiles_list2 must be provided when symmetric=False")
    
    if PROFILE:
        setup_start = time.perf_counter()
    
    # Compute filter2 bounds for all pairs using available bound functions (mirrors are_close_mols structure)
    if symmetric:
        bounds_results = mces_lower_bound_symmetric(smiles_list1)
    else:
        bounds_results = mces_lower_bound(smiles_list1, smiles_list2)  # type: ignore

    # Initialize distance matrix with bounds as initial estimates
    distance_matrix = bounds_results

    if PROFILE:
        setup_time = time.perf_counter() - setup_start
        print(f"Setup time: {setup_time:.3f}s")
        bounds_time = 0.0
        print(f"Bounds calculation done (shape {distance_matrix.shape})")


    selected_idx = np.argwhere(distance_matrix < threshold)
    if symmetric:
        # keep only i<j
        selected_idx = selected_idx[selected_idx[:, 0] < selected_idx[:, 1]]
        
    pairs_needing_exact: List[Tuple[int, int]] = [tuple(x) for x in selected_idx.tolist()]

    if PROFILE:
        print(f"Pairs needing exact calculation: {len(pairs_needing_exact)} out of {distance_matrix.size}")

    exact_results: List[Tuple[int, int, int]] = []
    if len(pairs_needing_exact) > 0:
        # delegate batching & parallelization to exact_mces_for_list_of_pairs
        exact_results = exact_mces_for_list_of_pairs(
            smiles_list1, smiles_list2, pairs_needing_exact,
            n_jobs=n_jobs, batch_size=batch_size, threshold=threshold, solver=solver
        )

    if PROFILE:
        print(f"Exact calculation completed, updating matrix")

    # Update distance matrix with exact results
    for i, j, distance in exact_results:
        distance_matrix[i, j] = distance
        if symmetric:
            distance_matrix[j, i] = distance

    if PROFILE:
        total_time = time.perf_counter() - total_start
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return distance_matrix

def are_close_mols(
    smiles_list1: List[str], 
    smiles_list2: Optional[List[str]] = None, 
    *, 
    n_jobs: int = -1, 
    batch_size: int = 20, 
    solver: str = "GUROBI",
    symmetric: bool = False) -> np.ndarray:
    """
    Efficiently computes whether each pair of molecules has an MCES distance of 1 or lower.

    First two arguments (smiles_list1, smiles_list2) can be passed positionally or by keyword.
    All other arguments are keyword-only.
    
    Parameters
    ----------
    smiles_list1 : list
    List of SMILES strings for the first set of molecules
    smiles_list2 : list or None
    List of SMILES strings for the second set of molecules.
    If None and symmetric=True, will compare molecules within smiles_list1.
    n_jobs : int
    Number of parallel jobs to run. -1 means use all available cores.
    symmetric : bool
    If True, optimize for comparing molecules within the same list.
    batch_size : int
    Number of pairs to process in each parallel job to reduce overhead.
    
    Returns
    -------
    numpy.ndarray   
    Boolean matrix where element [i,j] is True if molecules i and j have MCES distance ≤ 1
    """

    assert len(smiles_list1) > 0, "smiles_list1 is empty"
    if smiles_list2 is None:
        symmetric = True
    else:
        symmetric = False
        assert len(smiles_list2) > 0, "smiles_list2 is empty"
    if PROFILE:
        total_start = time.perf_counter()
        print("\n=== Profiling are_close_mols ===")
    
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Handle symmetric case
    if symmetric:
        smiles_list2 = smiles_list1

    if symmetric:
    # Calculate filter2 bounds for all pairs in symmetric case
        bounds_results = mces_lower_bound_symmetric(smiles_list1)
    else:
        bounds_results = mces_lower_bound(smiles_list1, smiles_list2) #type: ignore


    # Only perform expensive ILP computation on potential matches
    # Determine which pairs need the expensive ILP: bound <= 1

    # the bounds results are a square numpy array
    mask = bounds_results <= 1.0
    result_matrix = mask # if the bound is more than 1, then they are not close, now  we will iterate over anywhere the vbound is kess than 1 and fill in the truth
    
    selected = np.argwhere(mask)
    if selected.size > 0:
    #now, if it symmetric we remove all redundant pairs
        if symmetric:
            selected = selected[selected[:, 0] < selected[:, 1]]

        # Convert to list of tuples 
        pairs_needing_ilp = [tuple(x) for x in selected.tolist()]

        if PROFILE:
            print(f"Pairs needing ILP: {len(pairs_needing_ilp)} out of {len(bounds_results)}")
    
        # reuse exact_mces_for_list_of_pairs (performs batching + parallelism)
        if PROFILE:
            exact_start = time.perf_counter()
        
        exact_results = exact_mces_for_list_of_pairs(
            smiles_list1, smiles_list2, pairs_needing_ilp,
            n_jobs=n_jobs, batch_size=batch_size, threshold=2, solver=solver
        )

        if PROFILE:
            exact_time = time.perf_counter() - exact_start
            print(f"Exact calculation time: {exact_time:.3f}s")
            update_start = time.perf_counter()
        
        # Update result matrix
        if symmetric:
            for i, j, distance in exact_results:
                result_matrix[i, j] = False if distance is None else distance <= 1
                result_matrix[j, i] = result_matrix[i, j]
        else:
            for i, j, distance in exact_results:
                result_matrix[i, j] = False if distance is None else distance <= 1
        
        if PROFILE:
            update_time = time.perf_counter() - update_start
            print(f"Matrix update time: {update_time:.3f}s")
    
    if PROFILE:
        total_time = time.perf_counter() - total_start
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return result_matrix


def exact_mces_for_list_of_pairs(
    smiles_list1: List[str], 
    smiles_list2: List[str], 
    pairs: List[Tuple[int, int]], 
    n_jobs: int = -1, 
    batch_size: int = 20, 
    threshold: int = 10, 
    solver: str = "GUROBI"
) -> List[Tuple[int, int, int]]:
    """
    Efficiently computes exact MCES distances for a specific list of molecule pairs.
    
    Parameters
    ----------
    smiles_list1 : List[str]
        List of SMILES strings for the first set of molecules
    smiles_list2 : List[str]
        List of SMILES strings for the second set of molecules
    pairs : List[Tuple[int, int]]
        List of (i, j) index pairs to compute distances for
    n_jobs : int
        Number of parallel processes to run. -1 means use all available cores.
    batch_size : int
        Number of pairs to process in each batch to reduce overhead.
    threshold : int
        Distance threshold for early termination. -1 means no threshold.
    solver : str
        Solver to use for MCES calculation.
        
    Returns
    -------
    List[Tuple[int, int, int]]
        List of (i, j, distance) tuples for each requested pair
    """
    if PROFILE:
        total_start = time.perf_counter()
        print("\n=== Profiling exact_mces_for_list_of_pairs ===")
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    if PROFILE:
        setup_start = time.perf_counter()
    
    # Create batches of pairs to process together
    batches = list(batched(pairs, batch_size))
    
    if PROFILE:
        setup_time = time.perf_counter() - setup_start
        print(f"Setup time: {setup_time:.3f}s")
        print(f"Processing {len(pairs)} pairs in {len(batches)} batches")
        exact_start = time.perf_counter()
    
    # Use ProcessPoolExecutor for parallel processing
    all_results = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(_calculate_exact_batch, smiles_list1, smiles_list2, batch, threshold, solver): batch
            for batch in batches
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except Exception as e:
                batch = future_to_batch[future]
                print(f"Error processing batch {batch}: {e}")
                # Add None results for failed batch
                for i, j in batch:
                    all_results.append((i, j, None))
    
    if PROFILE:
        exact_time = time.perf_counter() - exact_start
        total_time = time.perf_counter() - total_start
        print(f"Exact calculation time: {exact_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print("=" * 40)
    
    return all_results



def construct_graph(smiles:str) -> nx.Graph:
    """ 
    Converts a SMILE into a graph
     
    Parameters
    ----------
    s : str 
        Smile of the molecule
        
    Returns:
    -------
    networkx.classes.graph.Graph
        Graph that represents the molecule.
        The bond types are represented as edge weights.
        The atom types are represented as atom attributes of the nodes.
    """
    #read the smile
    m: Chem.Mol = Chem.MolFromSmiles(smiles) # type :ignore
    # convert the molecule into a graph
    if m is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    # The bond and atom types are converted to node/edge attributes
    G: nx.Graph = nx.Graph()
    for atom in m.GetAtoms():
        G.add_node(atom.GetIdx(), atom=atom.GetSymbol())
    for bond in m.GetBonds():
        G.add_edge(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx(), weight=bond.GetBondTypeAsDouble())
    return G



def MCES_ILP(G1, G2, threshold, solver='default', solver_options={}, no_ilp_threshold=False):
    """
     Calculates the exact distance between two molecules using an ILP

     Parameters
     ----------
     G1 : networkx.classes.graph.Graph
         Graph representing the first molecule.
     G2 : networkx.classes.graph.Graph
         Graph representing the second molecule.
     threshold : float
         Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.
     solver: string
         ILP-solver used for solving MCES. Example:CPLEX_CMD
     solver_options: dict
         additional options to pass to solvers. Example: threads=1, msg=False for better multi-threaded performance
     no_ilp_threshold: bool
         if true, always return exact distance even if it is below the threshold (slower)

     Returns:
     -------
     float
         Distance between the molecules
     int
         Type of Distance:
             1 : Exact Distance
             2 : Lower bound (If the exact distance is above the threshold)

    """

    ILP=pulp.LpProblem("MCES", pulp.LpMinimize)

    #Variables for nodepairs
    nodepairs=[]
    for i in G1.nodes:
        for j in G2.nodes:
            if G1.nodes[i]["atom"]==G2.nodes[j]["atom"]:
                nodepairs.append(tuple([i,j]))
    y=pulp.LpVariable.dicts('nodepairs', nodepairs,
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)
    #variables for edgepairs and weight
    edgepairs=[]
    w={}
    for i in G1.edges:
        for j in G2.edges:
            if (G1.nodes[i[0]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[1]]["atom"]==G2.nodes[j[1]]["atom"]) or (G1.nodes[i[1]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[0]]["atom"]==G2.nodes[j[1]]["atom"]):
                edgepairs.append(tuple([i,j]))
                w[tuple([i,j])]=max(G1[i[0]][i[1]]["weight"],G2[j[0]][j[1]]["weight"])-min(G1[i[0]][i[1]]["weight"],G2[j[0]][j[1]]["weight"])

    #variables for not mapping an edge
    for i in G1.edges:
        edgepairs.append(tuple([i,-1]))
        w[tuple([i,-1])]=G1[i[0]][i[1]]["weight"]
    for j in G2.edges:
        edgepairs.append(tuple([-1,j]))
        w[tuple([-1,j])]=G2[j[0]][j[1]]["weight"]
    c=pulp.LpVariable.dicts('edgepairs', edgepairs,
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)


    #objective function
    ILP += pulp.lpSum([ w[i]*c[i] for i in edgepairs])

    #Every node in G1 can only be mapped to at most one in G2
    for i in G1.nodes:
        h=[]
        for j in G2.nodes:
            if G1.nodes[i]["atom"]==G2.nodes[j]["atom"]:
                h.append(tuple([i,j]))
        ILP+=pulp.lpSum([y[k] for k in h])<=1

    #Every node in G1 can only be mapped to at most one in G1
    for i in G2.nodes:
        h=[]
        for j in G1.nodes:
            if G1.nodes[j]["atom"]==G2.nodes[i]["atom"]:
                h.append(tuple([j,i]))
        ILP+=pulp.lpSum([y[k] for k in h])<=1

    #Every edge in G1 has to be mapped to an edge in G2 or the variable for not mapping has to be 1
    for i in G1.edges:
        ls=[]
        rs=[]
        for j in G2.edges:
            if (G1.nodes[i[0]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[1]]["atom"]==G2.nodes[j[1]]["atom"]) or (G1.nodes[i[1]]["atom"]==G2.nodes[j[0]]["atom"] and G1.nodes[i[0]]["atom"]==G2.nodes[j[1]]["atom"]):
                ls.append(tuple([i,j]))
        ILP+=pulp.lpSum([c[k] for k in ls])+c[tuple([i,-1])]==1

    #Every edge in G2 has to be mapped to an edge in G1 or the variable for not mapping has to be 1
    for i in G2.edges:
        ls=[]
        rs=[]
        for j in G1.edges:
            if (G1.nodes[j[0]]["atom"]==G2.nodes[i[0]]["atom"] and G1.nodes[j[1]]["atom"]==G2.nodes[i[1]]["atom"]) or (G1.nodes[j[1]]["atom"]==G2.nodes[i[0]]["atom"] and G1.nodes[j[0]]["atom"]==G2.nodes[i[1]]["atom"]):
                ls.append(tuple([j,i]))
        ILP+=pulp.lpSum([c[k] for k in ls])+c[tuple([-1,i])]==1

    #The mapping of the edges has to match the mapping of the nodes
    for i in G1.nodes:
        for j in G2.edges:
            ls=[]
            for k in G1.neighbors(i):
                if tuple([tuple([i,k]),j]) in c:
                    ls.append(tuple([tuple([i,k]),j]))
                else:
                    if  tuple([tuple([k,i]),j]) in c:
                        ls.append(tuple([tuple([k,i]),j]))
            rs=[]
            if G1.nodes[i]["atom"]==G2.nodes[j[0]]["atom"]:
                rs.append(tuple([i,j[0]]))
            if G1.nodes[i]["atom"]==G2.nodes[j[1]]["atom"]:
                rs.append(tuple([i,j[1]]))
            ILP+=pulp.lpSum([c[k] for k in ls])<=pulp.lpSum([y[k] for k in rs])


    for i in G2.nodes:
        for j in G1.edges:
            ls=[]
            for k in G2.neighbors(i):
                if tuple([j,tuple([i,k])]) in c:
                    ls.append(tuple([j,tuple([i,k])]))
                else:
                    if tuple([j,tuple([k,i])]) in c:
                        ls.append(tuple([j,tuple([k,i])]))
            rs=[]
            if G2.nodes[i]["atom"]==G1.nodes[j[0]]["atom"]:
                rs.append(tuple([j[0],i]))
            if G2.nodes[i]["atom"]==G1.nodes[j[1]]["atom"]:
                rs.append(tuple([j[1],i]))
            ILP+=pulp.lpSum([c[k] for k in ls])<=pulp.lpSum(y[k] for k in rs)

    #constraint for the threshold
    if threshold!=-1 and not no_ilp_threshold:
        ILP +=pulp.lpSum([ w[i]*c[i] for i in edgepairs])<=threshold

    #solve the ILP
    if solver.lower()=="default":
        sol= pulp.getSolver("PULP_CBC_CMD", msg=0,**solver_options)
        ILP.solve()
    elif solver.upper()=="GUROBI":
        # ILP.solve(pulp.GUROBI(**solver_options))
        sol:pulp.LpSolver=pulp.getSolver("GUROBI", **solver_options)
        ILP.solve(sol)
    elif solver.upper()=="CUOPT":
        ILP.solve(pulp.CUOPT(msg=0))
        print("CUOPT WAS USED")

    else:
        ILP.solve(pulp.PULP_CBC_CMD(msg=0,**solver_options))
    if ILP.status==1:
        val = ILP.objective.value()
        if val is None:
            return 0, 1
        return float(ILP.objective.value()),1
    else:
        return threshold,2


@contextmanager
def suppress_output() -> Generator[None, None, None]:
    """Suppress stdout and stderr output for both terminal and notebook environments"""
    # Save the original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    # Create dummy streams to redirect output
    # Using io.StringIO captures output in memory, os.devnull discards it.
    # os.devnull is generally better for pure suppression.
    devnull_w = open(os.devnull, 'w')
    # For notebooks, redirecting sys streams is usually sufficient.
    # File descriptor redirection can be problematic in notebooks.
    sys.stdout = devnull_w
    sys.stderr = devnull_w

    try:
        yield
    finally:
        # Restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Close the dummy stream
        devnull_w.close()

@lru_cache(maxsize=None)
def _cached_construct_graph(smiles: str) -> nx.Graph:
    return construct_graph(smiles)
# Process exact calculations in batches
def _calculate_exact_batch(smiles_list1: List[str], smiles_list2: List[str], batch_pairs: List[Tuple[int, int]], threshold: int, solver="default") -> List[Tuple[int, int, int]]:
    '''this function is used to calculate the exact MCES distance in batches.
    if the distance is greater than the threshold, it will return the threshold value.
    if the distance is less than the threshold, it will return the distance value.
    if the computation fails, it will return None.
    The function will return a list of tuples, where each tuple contains the indices of the two molecules and the distance value.
    '''
    with suppress_output():
        results = []
        for i, j in batch_pairs:
            # Load graphs on-demand
            g1: nx.Graph = _cached_construct_graph(smiles_list1[i])
            g2: nx.Graph = _cached_construct_graph(smiles_list2[j])
            try:
                distance, _ = MCES_ILP(g1, g2, threshold=threshold, solver=solver)
            except Exception as e:
                # Handle errors gracefully (including GurobiError)
                if "Gurobi" in str(type(e)) or "gurobi" in str(e).lower():
                    print(f"GurobiError for pair ({i}, {j}): {e}")
                else:
                    print(f"Error for pair ({i}, {j}): {e}")
                distance = None
            results.append((i, j, distance))
    return results






if __name__ == "__main__":
    from time import perf_counter
    nist_smiles: List[str] = pl.scan_parquet('/home/analytit_admin/dev/MS_encoder/data/NIST_prepared_labeled.parquet').sort('ExactMass').select('CanonicalSMILES').unique(maintain_order=True).collect().slice(offset=10000, length=2048).to_series().to_list()
    smiles1 : List[str] = nist_smiles[:1023]
    smiles2 : List[str] = nist_smiles[1024:]
    # smiles1 = nist_smiles

    # Example usage

    start = perf_counter()
    with suppress_output():
        result_matrix = calculate_mces_distances(smiles1, smiles2)
    print(f"Time taken: {perf_counter() - start:.2f} seconds")

    print(result_matrix)