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
