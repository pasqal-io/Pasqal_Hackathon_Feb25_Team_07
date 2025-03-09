import random

import cvxpy as cp
import numpy as np
import pandas as pd

# Adding versions of the problem for multi forecasting in series
NUM_SCENARIOS = 1000
NUM_GENS = 3

np.random.seed(42)


def initialize_parameters():

    INTITAL_PARAMETERS = {
        "quad_price": np.array([random.randint(1, 10) for _ in range(NUM_GENS)]),
        "linear_price": np.array([random.randint(1, 10) for _ in range(NUM_GENS)]),
        "constant_price": np.zeros(NUM_GENS),
        "p_max": np.array([random.randint(1, 30) for _ in range(NUM_GENS)]),
        "p_min": np.zeros(NUM_GENS),
        "gen_number": NUM_GENS,
    }

    total_available_capacity = sum(INTITAL_PARAMETERS["p_max"])

    INTITAL_PARAMETERS["total_demand"] = np.array(
        [random.randint(1, total_available_capacity)]
    )
    return INTITAL_PARAMETERS


def define_objective(params, P):
    return cp.Minimize(
        cp.sum(cp.multiply(params["quad_price"], cp.power(P, 2)))
        + cp.sum(cp.multiply(params["linear_price"], P))
        + cp.sum(params["constant_price"])
    )


def define_constraints(params, P, totalDemand):
    return [P >= params["p_min"], P <= params["p_max"], cp.sum(P) >= totalDemand]


def solve_unit_commitment(objective, constraints):
    problem = cp.Problem(objective, constraints)
    problem.solve()#solver=cp.ECOS_BB)
    return problem.status, problem.value


def print_results(status, value, generation):
    print("Status:", status)
    print("Optimal value:", value)
    print("Optimal solution:")
    for i, p in enumerate(generation):
        print(f"P{i+1} = {p:.4f}")
    print(f"Total generation: {sum(generation)}")


def main():

    rows = []

    for i in range(NUM_SCENARIOS):

        params = initialize_parameters()
        output_file = f"data/classical_{NUM_SCENARIOS}_scenarios_gens_{NUM_GENS}.csv"

        for j in range(len(params["total_demand"])):
            P = cp.Variable(params["gen_number"])
            objective = define_objective(params, P)
            constraints = define_constraints(params, P, params["total_demand"][j])
            status, value = solve_unit_commitment(objective, constraints)
            print_results(status, value, P.value)

            row = {"scenario": i, "total_demand": params["total_demand"][j]}
            # Add dynamic values for each generator
            for num in range(params["gen_number"]):
                row[f"quad_price_gen_{num}"] = params["quad_price"][num]
                row[f"linear_price_gen_{num}"] = params["linear_price"][num]
                row[f"Pmax_gen_{num}"] = params["p_max"][num]

            for num in range(params["gen_number"]):
                row[f"solution_gen_{num}"] = int(round(P.value[num], 3))

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file)


if __name__ == "__main__":
    main()
