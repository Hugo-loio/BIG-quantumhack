import numpy as np
from pulser import Register
from pulser.devices import DigitalAnalogDevice
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def evaluate_mapping(coords, *args):
    #Cost function to minimize. Ideally, the pairwise distances are conserved
    matrix, shape = args
    coords = np.reshape(coords, shape)
    approx_matrix = squareform( DigitalAnalogDevice.interaction_coeff / pdist(coords) ** 6)
    return np.linalg.norm(matrix - approx_matrix)

def optimized_coords(matrix, seed):
    shape = (len(matrix), 2)
    costs = []
    np.random.seed(seed)
    x0 = np.random.random(shape).flatten()
    bounds = [(0,35) for _ in x0]
    res = minimize(
        evaluate_mapping,
        x0,
        args=(matrix, shape),
        method="Nelder-Mead",
        tol=1e-8,
        bounds = bounds,
        options={"maxiter": 2000000, "maxfev": None},
    )
    return np.reshape(res.x, (len(matrix), 2))

def optimized_register(matrix, seed = 0):
    coords = optimized_coords(matrix, seed)
    qubits = dict(enumerate(coords))
    return Register(qubits)

def draw_register(register):
    register.draw(
        blockade_radius=DigitalAnalogDevice.rydberg_blockade_radius(1.0),
        draw_graph=False,
        draw_half_radius=True,
    )