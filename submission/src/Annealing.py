import numpy as np
from qutip import *
from scipy import *

def Annealing_GS(H1_matrix, N, M, taumax, t_steps):
    n = 1  # Size of each identity matrix
    k = N  # Number of identity matrices
    # Create a 1D array of identity matrices using dtype=object
    idd = 2*np.array([np.eye(n) for _ in range(k)], dtype=object)
    iddf = np.concatenate([mat.ravel() for mat in idd])
    iD=iddf.tolist()
    
    H1 = Qobj(H1_matrix,dims=[iD,iD])
    taulist = np.linspace(0, taumax, t_steps)
    # pre-allocate operators
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))


    psi_list = [basis(2,0) for n in range(N)]
    psi0 = tensor(psi_list)
    H0 = 0    
    for n in range(N):
        H0 += - 0.5 * 2.5 * sz_list[n]

    # the time-dependent hamiltonian in list-function format
    args = {'t_max': max(taulist)}
    h_t = [[H0, lambda t, args : (args['t_max']-t)/args['t_max']],
        [H1, lambda t, args : t/args['t_max']]]

    # callback function for each time-step
    #
    evals_mat = np.zeros((len(taulist),M))
    P_mat = np.zeros((len(taulist),M))
    idx = [0]
    ekets_ar = np.zeros((M,2**N)).astype('complex')
    def process_rho(tau, psi):
    
        # evaluate the Hamiltonian with gradually switched on interaction 
        H = qobj_list_evaluate(h_t, tau, args)

        # find the M lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=M)
        evals_mat[idx[0],:] = np.real(evals)
        eket_save = ekets
        for i in range(0,M):
            ekets_ar[i] = np.array((eket_save[i].full())[:,0], dtype=complex)
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2    
            
        idx[0] += 1
    qutip.mesolve(h_t, psi0, taulist, [], process_rho, args)
    return evals_mat, ekets_ar
