import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import src.register_optimizer as regop
import src.QAOA_func as QAOA_func
import src.qubo as qb


def QAOA(Q):
 
    def func(param, *args):
        Q = args[0]
        C = quantum_loop(param)
        cost = qb.get_cost(C, Q)
        return cost

    def quantum_loop(parameters):
        params = np.array(parameters)
        t_params, s_params = np.reshape(params.astype(int), (2, LAYERS))
        assigned_seq = seq.build(t_list=t_params, s_list=s_params)
        simul = QutipEmulator.from_sequence(assigned_seq, sampling_rate=0.01)
        results = simul.run()
        count_dict = results.sample_final_state()  # sample from the state vector
        return count_dict

    shape = (len(Q), 2)
  
    np.random.seed(0)
    
    reg = regop.optimized_register(Q)
    regop.draw_register(reg)

    LAYERS = 2

    # Parametrized sequence
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_global")

    t_list = seq.declare_variable("t_list", size=LAYERS)
    s_list = seq.declare_variable("s_list", size=LAYERS)

    for t, s in zip(t_list, s_list):
        pulse_1 = Pulse.ConstantPulse(1000 * t, 1.0, 0.0, 0)
        pulse_2 = Pulse.ConstantPulse(1000 * s, 0.0, 1.0, 0)

        seq.add(pulse_1, "ch0")
        seq.add(pulse_2, "ch0")

    seq.measure("ground-rydberg")


    scores = []
    params = []
    for repetition in range(20):
        guess = {
            "t": np.random.uniform(1, 10, LAYERS),
            "s": np.random.uniform(1, 10, LAYERS),
        }

        try:
            res = minimize(
                func,
                args=Q,
                x0=np.r_[guess["t"], guess["s"]],
                method="Nelder-Mead",
                tol=1e-5,
                options={"maxiter": 10},
            )
            scores.append(res.fun)
            params.append(res.x)
        except Exception as e:
            pass

    optimal_count_dict = quantum_loop(params[np.argmin(scores)])

    QAOA_func.plot_distribution(optimal_count_dict)