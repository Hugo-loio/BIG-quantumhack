import numpy as np
import math

#Number n of vertexes in QRAC:(n,1)
N_QRAC = 3

#-------------------------------------------------------------------------------
# Matrix definition
#-------------------------------------------------------------------------------
#Single spin Hilbert space dimension
d=2

# Define Pauli matrices
Sx = np.array([[0., 1.], [1., 0.]], dtype='cfloat') #single-site S^x
Sy = np.array([[0., -1j], [1j, 0.]], dtype='cfloat')  #single-site S^y
Sz = np.array([[1., 0.], [0., -1.]], dtype='cfloat')  #single-site S^z
# S_op contains all of them
S_op = np.zeros([3,2,2], dtype='cfloat')
S_op[0] = Sz
S_op[1] = Sy
S_op[2] = Sx

# This function builds the operator:
# ---> Op = Si_n Sj_m
# ---> where i,j=(x,y,z) and n,m=(1,2,...,N)
def H2(Si, Sj, n, m, N):
    if m>n:
        op = np.identity(d**(n-1), dtype='cfloat')
        op = np.kron(op, Si)
        op = np.kron(op, np.identity(d**(m-n-1), dtype='cfloat'))
        op = np.kron(op, Sj)
        op = np.kron(op, np.identity(d**(N-m), dtype='cfloat'))
    elif n>m:
        op = np.identity(d**(m-1), dtype='cfloat')
        op = np.kron(op, Sj)
        op = np.kron(op, np.identity(d**(n-m-1), dtype='cfloat'))
        op = np.kron(op, Si)
        op = np.kron(op, np.identity(d**(N-n), dtype='cfloat'))
    elif m==n:
        op = np.identity(d**(n-1), dtype='cfloat')
        op = np.kron(op, Si*Sj)
        op = np.kron(op, np.identity(d**(N-n), dtype='cfloat'))
    return op

# This function builds the operator:
# ---> Op = Si_n 
# ---> where i=(x,y,z) and n=(1,2,...,N)
def H1(Si, i, N):
    op = np.identity(d**(i-1), dtype='cfloat')
    op = np.kron(op, Si)
    op = np.kron(op, np.identity(d**(N-i), dtype='cfloat'))
    return op

# Define the Hamiltonian of the system
# ---> N = number of sites
# ---> param = array with parameters of Hamiltonian
def H(N, param):
    op = 0.0
    for i in range(1, N, 1):
        op += +param[0]*H2(Sx, Sx, i, i+1, N) 
    op += param[1]*H1(Sy, 1, N)
    op += param[2]*H1(Sz, N, N)
    return  op

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#-------------------------------------------------------------------------------
# Vertex coloring function using LDF method
#-------------------------------------------------------------------------------
def large_degree_first_coloring(adj_matrix):
    n = len(adj_matrix)
    
    # Step 1: Calculate the degree of each vertex
    degrees = [sum(adj_matrix[i]) for i in range(n)]
    
    # Step 2: Sort vertices by descending degree
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Step 3: Initialize color assignment
    color_assignment = [-1] * n
    
    # Step 4: Assign colors to each vertex
    for vertex in sorted_vertices:
        # Find colors used by adjacent vertices
        used_colors = set()
        for neighbor in range(n):
            if adj_matrix[vertex][neighbor] == 1 and color_assignment[neighbor] != -1:
                used_colors.add(color_assignment[neighbor])
        
        # Assign the smallest available color
        color = 0
        while color in used_colors:
            color += 1
        color_assignment[vertex] = color

    return color_assignment

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def features_def(color, n_v, n_col):
    # Feature is a 3xN_v matrix:
    # Feature[0]=Colors of each vertex
    # Feature[1]=Number of the qubit of a given color
    # Feature[2]=Operator (X,Y,Z)=(0,1,2) associated to each vertex
    features = np.zeros((3, n_v))
    features[0] = color
    for i in range(0, n_col):
        qbit_numb = 0
        for j in range(0, n_v):
            if features[0,j]==i:
                features[1,j] = qbit_numb//3
                features[2,j] = qbit_numb%3
                qbit_numb+=1
    return features


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def QRAC_HAMILTONIAN(matrix, N_v):
    #-------------------------------------------------------------------------------
    #Setting graph colors and define qubits
    #-------------------------------------------------------------------------------
    # Array with colors
    n_colors = 0
    coloring = np.array(large_degree_first_coloring(matrix))
    n_colors = np.max(coloring)+1

    # We associate each vertex to a qubit
    # Feature is a 3xN_v matrix:
    # Feature[0]=Colors of each vertex
    # Feature[1]=Number of the qubit of a given color
    # Feature[2]=Operator (X,Y,Z)=(0,1,2) associated to each vertex
    features = features_def(coloring, N_v, n_colors).astype('int')
    print('--------------------')
    print('FEATURES=\n',features)
    #-------------------------------------------------------------------------------
    #Create H_relax
    #-------------------------------------------------------------------------------
    # Count the number of Qubits
    N_qubit = 0
    Qubit_ordering = np.zeros(N_v)
    for c in range(0, n_colors):
        #N_qubit += np.sum(np.where(features[0]==c, 1, 0))//3+np.sum(np.where(features[0]==c, 1, 0))%3
        N_qubit += math.ceil(np.sum(np.where(features[0]==c, 1, 0))/3)
    print('--------------------')
    print('Number of QBITS=',N_qubit)

    # Qbit_ordering is a vector that for every vertex determines the position of associated qubit in the chain
    # ATTENTION: elements in the chain are starting from 1: [1,2,...,N]
    for c in range(n_colors):
        color_indx = np.argwhere(features[0]==c)
        n_color_indx = len(color_indx)
        Qubit_ordering[color_indx]+=np.max(Qubit_ordering)
        for j in range(0,n_color_indx):
            Qubit_ordering[color_indx[j]] += 1+(j)//3
    Qubit_ordering = Qubit_ordering.astype(int)
    print('--------------------')
    print('Qbits ordering=',Qubit_ordering)

    # In the spin chain we order qbits for colors
    # S_op[0] = Sx, S_op[1] = Sy, S_op[2] = Sz
    H_RELAX = np.zeros((2**N_qubit, 2**N_qubit), dtype='cfloat')
    for i in range(0, N_v):
        for j in range(0, N_v):
            H_RELAX += matrix[i,j]*H2(S_op[features[2,i]], S_op[features[2,j]], Qubit_ordering[i], Qubit_ordering[j], N_qubit)
    H_RELAX = 0.5*(np.identity(d**(N_qubit), dtype='cfloat') - 3*H_RELAX)
    print('--------------------')
    print('H_RELAX=\n',H_RELAX)
    return H_RELAX, N_qubit, Qubit_ordering, features 

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Decode the graph and find the resulting partition
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Define Pauli matrices
E1 = np.zeros([2,2,2], dtype='cfloat')
E2 = np.zeros([2,2,2], dtype='cfloat')
E3 = np.zeros([2,2,2], dtype='cfloat')
St_p = 1/(np.sqrt(2))*np.array([1.,1.], dtype='cfloat')
St_m = 1/(np.sqrt(2))*np.array([1.,-1.], dtype='cfloat')
St_pi = 1/(np.sqrt(2))*np.array([1.,1j], dtype='cfloat')
St_mi = 1/(np.sqrt(2))*np.array([1.,-1j], dtype='cfloat')
St_0 = np.array([1.,0.], dtype='cfloat')
St_1 = np.array([0.,1.], dtype='cfloat')
E1[0] = St_p[:,np.newaxis]*St_p[np.newaxis,:]
E1[1] = St_m[:,np.newaxis]*St_m[np.newaxis,:]
E2[0] = St_pi[:,np.newaxis]*St_pi[np.newaxis,:]
E2[1] = St_mi[:,np.newaxis]*St_mi[np.newaxis,:]
E3[0] = St_0[:,np.newaxis]*St_0[np.newaxis,:]
E3[1] = St_1[:,np.newaxis]*St_1[np.newaxis,:]

E_tot = np.zeros([3,2,2,2], dtype='cfloat')
E_tot[0] = E1
E_tot[1] = E2
E_tot[2] = E3

def Graph_partition_func(gs_vector, N_v, qbit_ord, n_qbit, features):

    St_partition_0 = np.zeros(N_v).astype('complex')
    St_partition_1 = np.zeros(N_v).astype('complex')

    for i in range(0, N_v):
        St_partition_0[i] = (np.einsum('i,i->',np.einsum('i,ij->j',np.conjugate(gs_vector),H1(E_tot[features[2,i],0],qbit_ord[i],n_qbit)),gs_vector))
        St_partition_1[i] = (np.einsum('i,i->',np.einsum('i,ij->j',np.conjugate(gs_vector),H1(E_tot[features[2,i],1],qbit_ord[i],n_qbit)),gs_vector))

    St_partition_0 = np.real(St_partition_0)
    St_partition_1 = np.real(St_partition_1)

    GRAPH_PARTITION_0 = np.where(St_partition_0>0, 1, 0)
    GRAPH_PARTITION_1 = np.where(St_partition_1>0, 1, 0)
    return GRAPH_PARTITION_0, GRAPH_PARTITION_1
