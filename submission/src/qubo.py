import numpy as np
from qubovert import QUBO

def flattened_edges(graph):
    edges = graph.edges()
    nodes = list(graph.nodes())
    flattened_edges = []
    for (node1, node2) in edges:
        flattened_edges.append((nodes.index(node1), nodes.index(node2)))
    return(flattened_edges)

def QUBO_Q(graph):
    dim = len(list(graph.nodes()))
    Q = np.zeros((dim,dim))
    edges = flattened_edges(graph)
    for i,j in edges:
        Q[i,i] -= 1
        Q[j,j] -= 1
        Q[i,j] += 1
        Q[j,i] += 1
    return Q

def graph_matrix(graph):
    edges = flattened_edges(graph)
    dim = len(list(graph.nodes()))
    matrix = np.zeros((dim,dim))
    for i,j in edges:
        matrix[i,j] = 1
        matrix[j,i] = 1
    return matrix

def QUBO_solver(Q):
    qubo = QUBO()
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if Q[i, j] != 0:
                qubo[(i, j)] = Q[i, j]
    solution = qubo.solve_bruteforce()
    return np.array([val for val in solution.values()])
    #return []


def get_cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost


def get_cost(counter, Q):
    cost = sum(counter[key] * get_cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples