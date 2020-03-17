import numpy as np
from operator import itemgetter

# function [w_st, ST, X_st] = kruskal(X, w)
#
# This function finds the minimum spanning tree of the graph where each
# edge has a specified weight using the Kruskal's algorithm.
#
# Assumptions
# -----------
#     N:  1x1  scalar      -  Number of nodes (vertices) of the graph
#    Ne:  1x1  scalar      -  Number of edges of the graph
#   Nst:  1x1  scalar      -  Number of edges of the minimum spanning tree
#
# We further assume that the graph is labeled consecutively. That is, if
# there are N nodes, then nodes will be labeled from 1 to N.
#
# INPUT
#
#     X:  NxN logical      -  Adjacency matrix
#             matrix          If X(i,j)=1, this means there is directed edge
#                             starting from node i and ending in node j.
#                             Each element takes values 0 or 1.
#                             If X symmetric, graph is undirected.
#
#  or     Nex2 double      -  Neighbors' matrix
#              matrix         Each row represents an edge.
#                             Column 1 indicates the source node, while
#                             column 2 the target node.
#
#     w:  NxN double       -  Weight matrix in adjacency form
#             matrix          If X symmetric (undirected graph), w has to
#                             be symmetric.
#
#  or     Nex1 double      -  Weight matrix in neighbors' form
#              matrix         Each element represents the weight of that
#                             edge.
#
#
# OUTPUT
#
#  w_st:    1x1 scalar     -  Total weight of minimum spanning tree
#    ST:  Nstx2 double     -  Neighbors' matrix of minimum spanning tree
#               matrix
#  X_st:  NstxNst logical  -  Adjacency matrix of minimum spanning tree
#                 matrix      If X_st symmetric, tree is undirected.
#
# EXAMPLES
#
# Undirected graph
# ----------------
# Assume the undirected graph with adjacency matrix X and weights w:
#
#         1
#       /   \
#      2     3
#     / \
#    4 - 5
#
# X = [0 1 1 0 0;
#      1 0 0 1 1;
#      1 0 0 0 0;
#      0 1 0 0 1;
#      0 1 0 1 0];
#
# w = [0 1 2 0 0;
#      1 0 0 2 1;
#      2 0 0 0 0;
#      0 2 0 0 3;
#      0 1 0 3 0];
#
# [w_st, ST, X_st] = kruskal(X, w);
# The above function gives us the minimum spanning tree.
#
#
# Directed graph
# ----------------
# Assume the directed graph with adjacency matrix X and weights w:
#
#           1
#        / ^ \
#       / /   \
#      v       v
#       2 ---> 3
#
# X = [0 1 1
#      1 0 1
#      0 0 0];
#
# w = [0 1 4;
#      2 0 1;
#      0 0 0];
#
# [w_st, ST, X_st] = kruskal(X, w);
# The above function gives us the minimum directed spanning tree.
#
#

def cnvrtX2ne(X, isUndirGraph):
    if isUndirGraph:
        ne = np.zeros()

    return ne

def cnvrtw2ne(w, ne):
    return w

def kruskal(X, w):
    print("********************** Kruskal MST **********************")
    isUndirGraph = 1

    # print((X[:]==0).sum())
    # print((X[:]==1).sum())
    #print(len(X)*len(X[0]))
    #print((X==0).sum())

    if X.shape[0] == X.shape[1] and (X[:]==1).sum()==len(X)*len(X[0]):
        if (X-X.transpose()).any().any():
            isUndirGraph = 0
        ne = cnvrtX2ne(X, isUndirGraph)
    else:
        #size(unique(sort(X,2),'rows'),1)~=size(X,1)
        X.sort(axis=1)
        for i in range(len(X)):
            print(X[i])
        X = X.tolist()
        #print(X)
        tmp = list(set([tuple(ti) for ti in X]))
        #print(tmp)
        tmp.sort()
        #print(tmp)
        tmp_len = len(tmp)
        #print(len(X))
        if tmp_len != len(X):
            isUndirGraph = 0

        ne = X

        return 1,1,1





    # print(X.shape[0])

    # Convert logical adjacent matrix to neighbors' matrix
    return 1,1,1
