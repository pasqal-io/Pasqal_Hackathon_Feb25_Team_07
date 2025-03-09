import sympy
import pyqubo
import neal
import config_ed_uc
import numpy as np
import os
import time
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import csv
from matplotlib.patches import Patch


def initialize_parameters():
    return {
        'quad_price': config_ed_uc.get_parameter('quad_price'),
        'linear_price': config_ed_uc.get_parameter('linear_price'),
        'constant_price': config_ed_uc.get_parameter('constant_price'),
        'genNumber': config_ed_uc.get_parameter('genNumber'),
        'totalDemand': config_ed_uc.get_parameter('totalDemand'),
        'penaltyDemand': config_ed_uc.get_parameter('penaltyDemand'),
        'penaltyRange': config_ed_uc.get_parameter('penaltyRange'),
        'Pmax': config_ed_uc.get_parameter('Pmax'),
        'Pmin': config_ed_uc.get_parameter('Pmin'),
        'num_reads_dwave': config_ed_uc.get_parameter('num_reads_dwave')
    }

def calculate_binary_expansion_size(params):
    return [
        (params['Pmax'][i] - params['Pmin'][i]).bit_length()
        for i in range(params['genNumber'])
    ]

def create_binary_variables(params, binary_expansion_size, scenario_number):
    binary_vars = []
    y_binary_vars = []
    for i in range(params['genNumber']):
        gen_vars = []
        y_gen_vars = []
        for bit in range(binary_expansion_size[i]):
            scaling_factor = 2 ** bit
            x_var = pyqubo.Binary(f'x_{i}_{bit}_{scenario_number}')
            y_var = sympy.symbols(f'x_{i}_{bit}_{scenario_number}')
            gen_vars.append((scaling_factor, x_var))
            y_gen_vars.append((scaling_factor, y_var))
        binary_vars.append(gen_vars)
        y_binary_vars.append(y_gen_vars)
    return binary_vars, y_binary_vars

def build_objective_qubo(params, binary_vars, y_binary_vars):
    qubo = 0
    y_qubo = 0
    for i in range(params['genNumber']):
        alpha, beta = params['quad_price'][i], params['linear_price'][i]
        full_expansion = sum(sf * x for sf, x in binary_vars[i])
        y_full_expansion = sum(sf * y for sf, y in y_binary_vars[i])
        
        qubo += alpha * (full_expansion ** 2) + beta * full_expansion
        y_qubo += alpha * (y_full_expansion ** 2) + beta * y_full_expansion
    
    total_constant_price = sum(params['constant_price'])
    return qubo + total_constant_price, y_qubo + total_constant_price

def create_slack_variables(params, binary_expansion_size, scenario_number):
    slack_needed = [
        params['Pmax'][i] - params['Pmin'][i]
        for i in range(params['genNumber'])
    ]
    
    slack_vars = []
    y_slack_vars = []
    for i, needed in enumerate(slack_needed):
        s_vars = []
        y_s_vars = []
        for bit in range(needed.bit_length()):
            scaling_factor = 2 ** bit
            s_var = pyqubo.Binary(f's_{i}_{bit}_{scenario_number}')
            y_s_var = sympy.symbols(f's_{i}_{bit}_{scenario_number}')   
            s_vars.append((scaling_factor, s_var))
            y_s_vars.append((scaling_factor, y_s_var))
        slack_vars.append(s_vars)
        y_slack_vars.append(y_s_vars)
    return slack_vars, y_slack_vars

def add_constraints(params, qubo, y_qubo, binary_vars, y_binary_vars, slack_vars, y_slack_vars, total_demand):
    for i in range(params['genNumber']):
        binary_sum = sum(sf * x for sf, x in binary_vars[i])
        slack_sum = sum(sf * s for sf, s in slack_vars[i])
        max_constraint = params['penaltyRange'] * (params['Pmax'][i] - binary_sum - slack_sum)**2
        qubo += max_constraint #+ params['penaltySlack'] * slack_sum

        y_binary_sum = sum(sf * y for sf, y in y_binary_vars[i])
        y_slack_sum = sum(sf * y_s for sf, y_s in y_slack_vars[i])
        y_max_constraint = params['penaltyRange'] * (params['Pmax'][i] - y_binary_sum - y_slack_sum)**2
        y_qubo += y_max_constraint #+ params['penaltySlack'] * y_slack_sum

    total_binary_sum = sum(sf * x for gen in binary_vars for sf, x in gen)
    y_total_binary_sum = sum(sf * y for gen in y_binary_vars for sf, y in gen)
    
    qubo += params['penaltyDemand'] * (total_demand - total_binary_sum)**2
    y_qubo += params['penaltyDemand'] * (total_demand - y_total_binary_sum)**2
    
    return qubo, y_qubo

def solve_qubo_sim(model, bqm, params):
    time_start = time.time()
    sa = neal.SimulatedAnnealingSampler()

    sampleset = sa.sample(bqm, num_reads=params['num_reads_dwave'])
    print(f'Execution time Simulated Annealing: {time.time() - time_start:.4f} s')
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample_sim = min(decoded_samples, key=lambda x: x.energy)
    solution_str = ''.join(str(best_sample_sim.sample[k]) for k in sorted(best_sample_sim.sample.keys()))
    print("Best solution as bitstring:", solution_str)
    return best_sample_sim, solution_str

def compile_qubo(qubo):
    model = qubo.compile()
    bqm = model.to_bqm()
    return model, bqm

def matrix_qubo(y_qubo): # To use the function the qubo must be written in sympy format

    y_qubo_expanded = sympy.expand(y_qubo)

    print("Expanded y_qubo:")
    print(y_qubo_expanded)

    # Get a sorted list of all symbols that appear.
    vars_list = sorted(y_qubo_expanded.free_symbols, key=lambda s: s.name)
    num_vars = len(vars_list)

    print("Identified variables:")
    for s in vars_list:
        print(s)

    print(f'Number of variables: {num_vars}')

    # Initialize the QUBO coefficient matrix.
    Q = np.zeros((num_vars, num_vars))

    # Convert the expanded expression into a polynomial in these symbols.
    poly_exp = sympy.Poly(y_qubo_expanded, vars_list)

    # Iterate over all terms (monomials) in the polynomial.
    for monom, coeff in poly_exp.terms():
        coeff = float(coeff)
        # monom is a tuple of exponents corresponding to each variable
        # Count the degree of the monomial.
        degree = sum(monom)
        if degree == 0:
            # Constant term can be dropped or stored separately.
            continue
        elif degree == 1:
            # Find the variable corresponding to the nonzero exponent.
            for idx, exp in enumerate(monom):
                if exp != 0:
                    Q[idx, idx] += coeff
        elif degree == 2:
            # Find the two indices or a squared term.
            indices = []
            for idx, exp in enumerate(monom):
                for _ in range(exp):
                    indices.append(idx)
            if len(indices) != 2:
                raise ValueError("Unexpected degree distribution for quadratic term.")
            i, j = indices
            if i == j:
                Q[i, i] += coeff
            else:
                # Distribute the coefficient equally; note that in a QUBO
                # the cost is written as xᵀQx with off-diagonals contributing twice.
                Q[i, j] += coeff / 2
                Q[j, i] += coeff / 2
        else:
            # For higher-order terms one must perform a reduction to quadratic
            # (for example via standard quadratization techniques).
            print("Found higher order term of degree", degree, "which is not handled automatically:", monom, "with coefficient", coeff)

    Q = Q
    num_zero_cells = np.sum(Q == 0)
    total_cells = Q.size
    percentage_zero_cells = (num_zero_cells / total_cells) * 100
    print(f"Number of zero cells in the QUBO matrix: {num_zero_cells}")
    print(f"Percentage of zero cells in the QUBO matrix: {percentage_zero_cells:.2f}%")
    return Q

def normalize_qubo_matrix(Q: np.ndarray) -> np.ndarray:
        """
        Normalizes the QUBO matrix so that its values are scaled between -1 and 1.

        Args:
            Q (np.ndarray): The original QUBO matrix.

        Returns:
            np.ndarray: The normalized QUBO matrix.
        """
        max_abs = np.max(np.abs(Q))
        if max_abs == 0:
            return Q
        
        return Q / max_abs

def matrix_diagonal(Q):
    diag_mean = np.mean(np.diag(Q))
    np.fill_diagonal(Q, diag_mean)
    return Q

def verifiy_pulser(Q):
    bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
    costs = []
    # this takes exponential time with the dimension of the QUBO
    for b in bitstrings:
        z = np.array(list(b), dtype=int)
        cost = z.T @ Q @ z
        costs.append(cost)
    zipped = zip(bitstrings, costs)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    return sort_zipped[:3]

def pulser_embedding(Q):
    def evaluate_mapping(new_coords, Q):
        """Cost function to minimize. Ideally, the pairwise distances are conserved."""
        new_coords = np.reshape(new_coords, (len(Q), 2))
        # computing the matrix of the distances between all coordinate pairs
        new_Q = squareform(
            DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6 
        )
        return np.linalg.norm(new_Q - Q)

    costs = []
    np.random.seed(0)
    x0 = np.random.random(len(Q) * 2)
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q,),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )
    coords = np.reshape(res.x, (len(Q), 2))

     # Scale and re-center coordinates if atoms are too far from the center.
    center = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    max_distance = np.max(distances)
    allowed_radius = 50  # in μm
    if max_distance > allowed_radius:
        scale_factor = allowed_radius / max_distance
        coords = center + scale_factor * (coords - center)
    coords = coords - np.mean(coords, axis=0)



    qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}
    reg = Register(qubits)
    reg.draw(
        blockade_radius=DigitalAnalogDevice.rydberg_blockade_radius(1.0),
        draw_graph=False,
        draw_half_radius=True,
    )

    return reg

def run_pulser(Q, reg, sort_zipped):
    # We choose a median value between the min and the max
    Omega = np.median(Q[Q > 0].flatten())
     # Cap Omega to not exceed the device limit (adjust max_amplitude as needed).
    max_amplitude = 1.0
    Omega = min(Omega, max_amplitude)
    delta_0 = -5  # just has to be negative
    delta_f = -delta_0  # just has to be positive
    T = 4000  # time in ns, we choose a time long enough to ensure the propagation of information in the system

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )

    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    seq.draw()

    start_time = time.time()
    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()
    print(f"Execution time Pulser: {time.time() - start_time:.2f} s")  
    final = results.get_final_state()
    count_dict = results.sample_final_state()
    print(f'printing the dictionary of the counts output:', count_dict)

    top_counts = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True)[:5])
    plot_distribution(top_counts, sort_zipped)
    return top_counts

def plot_distribution(C, sort_zipped):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = ['0101000001']  # QUBO solutions
    color_dict = {key: "g" if key in indexes else "r" for key in C}
    print(f'Optimal solutions', sort_zipped[:3])
    plt.figure(figsize=(12, 6))
    plt.title("QUBO Bitstring Distribution", fontsize=16)
    legend_elements = [
        Patch(facecolor="g", label="Optimal"),
        Patch(facecolor="r", label="Infeasible"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show() 

def main():
    params = initialize_parameters()
    for i in range(len(params['totalDemand'])): 
        
        binary_expansion_size = calculate_binary_expansion_size(params)
        binary_vars, y_binary_vars = create_binary_variables(params, binary_expansion_size, i)
        qubo, y_qubo = build_objective_qubo(params, binary_vars, y_binary_vars)
        slack_vars, y_slack_vars = create_slack_variables(params, binary_expansion_size, i)
        qubo, y_qubo = add_constraints(params, qubo, y_qubo, binary_vars, y_binary_vars, slack_vars, y_slack_vars, params['totalDemand'][i])
        if i == 0:
            total_qubo = qubo
            total_y_qubo = y_qubo
        else:
            total_qubo += qubo
            total_y_qubo += y_qubo
    model, bqm = compile_qubo(total_qubo)
    best_sample_sim, solution_string = solve_qubo_sim(model, bqm, params)
    Q_raw = matrix_qubo(total_y_qubo) ### I need to make sure that the output matrix is in the right format where I know which columns refer to which variable
    Q = matrix_diagonal(Q_raw)

    ### PULSER
    reg = pulser_embedding(Q)
    sort_zipped = verifiy_pulser(Q)
    top_counts = run_pulser(Q, reg, sort_zipped)
    # Define the output CSV path
    output_csv = os.path.join(os.path.dirname(__file__), 'results_with_params.csv')

    # Prepare rows to write
    rows = []

    # Write a header for the parameters section
    rows.append(["Test Parameters", "Value"])
    for key, value in params.items():
        rows.append([key, str(value)])
    rows.append([])  # empty row for separation

    # Header for the bitstring data
    rows.append(["Bitstring", "Last 5 Bits", "Count"])

    # Process each bitstring in top_counts, extracting the last 5 bits
    for bitstring, count in top_counts.items():
        last_five = bitstring[-5:]
        rows.append([bitstring, last_five, count])

    # Write rows to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Results saved to {output_csv}")

    print(f"QUBO as matrix {Q}")
    output_path = os.path.join(os.path.dirname(__file__), 'qubo_matrix_output/Q.csv')
    np.savetxt(output_path, Q, delimiter=',')
    print(f"QUBO with generation constraints {total_y_qubo}")
    print("Best sample configuration using sim:", best_sample_sim.sample)
    print(f"Energy with sim: {best_sample_sim.energy}")

    #QUBO
    print(f'QUBO sympy:', total_y_qubo)

if __name__ == "__main__":
    main()
