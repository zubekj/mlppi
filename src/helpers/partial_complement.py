# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 07:56:35 2014

@author: Mnich
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse.csgraph import shortest_path

def create_partial_complement(list_of_edges, verbose=False):
    #wierzchołki, pojedynczo
    nodes = np.unique([node for edge in list_of_edges for node in edge])
    N = len(nodes)

    #stworzenie macierzy reprezentującej graf
    G1 = np.zeros((N,N))
    for edge in list_of_edges:
        i0, i1 = np.where(nodes==edge[0]), np.where(nodes==edge[1])
        G1[i0,i1], G1[i1,i0] = 1, 1

    #stworzenie macierzy odległości
    distance_matrix = shortest_path(G1,directed=False,unweighted=True,overwrite=False)
    G1=np.where(G1==1,1,0) #shortest_path konwertowało 0 na inf

    #obliczenie stopni
    degrees = np.sum(G1,axis=1)

    #heurystyka
    G2 = np.zeros((N,N))
    G1 += np.identity(N, dtype=np.int64) #zabezpieczenie przed wybraniem tego samego wierzchołka
    garbage_degrees = np.zeros((N,)) #niewykorzystane stopnie
    new_edges = []
    while np.sum(degrees)>0:
        start_node =np.random.choice(np.where(degrees==np.amax(degrees))[0])  #losowy wybor wierzcholka, stopien zawsze > 0
        candidates = G1[start_node] + G2[start_node]#kandydaci mają tu wartość zero
        candidates_degrees = np.where(candidates==0, degrees, 0)
        largest_degree = np.amax(candidates_degrees)
        if largest_degree > 0: #tj. jeśli istnieją dostępne wierzchołki, dla obecnie wybranego
            candidates_distances = np.where(candidates_degrees == largest_degree, distance_matrix[start_node], 0)
            end_node = np.random.choice(np.where(candidates_distances == np.amax(candidates_distances))[0])
            G2[start_node,end_node], G2[end_node,start_node] = 1, 1
            new_edges.append((nodes[start_node],nodes[end_node]))
            degrees[end_node] -= 1
            degrees[start_node] -= 1
        else:
            garbage_degrees[start_node] = degrees[start_node]
            degrees[start_node] = 0 #wyrzucenie wierzchołka, ktry nie ma się z kim połączyć

    if verbose:
        print "unused degrees:", {node: degree for node, degree in zip(nodes, garbage_degrees)
                if degree > 0}

    return new_edges


# testing code
if __name__ == "__main__":

    def calc_degrees(l):
        nodes = defaultdict(int)
        for e in l:
            for v in e:
                nodes[v] += 1
        return [(n, nodes[n]) for n in sorted(nodes.keys())]

    g1 = pd.read_csv("test_interactions.csv").values
    d1 = calc_degrees(g1)
    g2 = create_partial_complement(g1)
    d2 = calc_degrees(g2)

    pos_neg_graph = pd.DataFrame(np.vstack((g1, g2)))
    pos_neg_graph[2] = 0
    pos_neg_graph[2][:g1.shape[0]] = 1

    pos_neg_graph.to_csv("pos_neg_interactions.csv", index=False)


