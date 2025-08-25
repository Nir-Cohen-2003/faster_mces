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
from .graph_construction import construct_graph
from .bounds import mces_lower_bound, mces_lower_bound_symmetric
from functools import lru_cache

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
        # create batches and run exact computation in parallel
        exact_batches = list(batched(pairs_needing_exact, batch_size))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_batch = {
                executor.submit(_calculate_exact_batch, smiles_list1, smiles_list2, batch, threshold, solver): batch for batch in exact_batches
            }
            exact_batch_results = []
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    exact_batch_results.append(future.result())
                except Exception as e:
                    print(f"Error processing exact batch {batch}: {e}")
                    fallback = [(i, j, None) for i, j in batch]
                    exact_batch_results.append(fallback)
            exact_results = [item for sublist in exact_batch_results for item in sublist]

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
    
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:

            if PROFILE:
                exact_start = time.perf_counter()
            
            # Create batches for ILP computation
            ilp_batches = batched(pairs_needing_ilp, batch_size)
            
            # Calculate exact results in batches - pass SMILES lists instead of precomputed graphs
            future_to_batch = {
            executor.submit(_calculate_exact_batch, smiles_list1, smiles_list2, batch, 2, solver): batch for batch in ilp_batches
            }
            exact_batch_results = []
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
            try:
                exact_batch_results.append(future.result())
            except Exception as e:
                print(f"Error processing exact batch {batch}: {e}")
                fallback = [(i, j, None) for i, j in batch]
                exact_batch_results.append(fallback)
            
            # Flatten batch results
            exact_results = list(chain(*exact_batch_results))
            
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
    threshold: int = -1, 
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
    # with suppress_output():
    result_matrix= are_very_distinct(smiles1, symmetric=True,batch_size=200,solver="GUROBI")
    print(f"Time taken: {perf_counter() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = are_very_distinct(smiles1,symmetric=False)
    # print(f"Time taken: {time() - start:.2f} seconds")
    # start = time()
    # with suppress_output():
    #     result_matrix = calculate_mces_distances(smiles1, smiles2)
    # print(f"Time taken: {time() - start:.2f} seconds")

    # print(result_matrix)