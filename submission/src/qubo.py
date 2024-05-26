import numpy as np
from qubovert import QUBO

# TODO: Check networkx
def QUBO_Q(graph):
    dim = len(graph.vertices)
    Q = np.zeros((dim,dim))
    for i,j in graph.edges:
        Q[i,i] += 1
        Q[j,j] += 1
        Q[i,j] -= 2
    return Q

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