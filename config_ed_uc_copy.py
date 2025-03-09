# config.py

# Define initial parameters 
INITIAL_PARAMETERS = {
        'quad_price': [1, 1],
        'linear_price': [4, 3],
        'constant_price': [0,0],
        'genNumber': 2, # Make sure this number is the same as length of prices
        'totalDemand': [4], #, 14, 16, 13, 17],
        'penaltyDemand': 100, # For 3 bus this is 10 for 1 and 2 100
        'penaltyRange': 100,# same on this
        'Pmax': [3, 5],
        'Pmin': [0, 0],
        'num_reads_dwave': 1000
    }

# Function to get a parameter value
def get_parameter(param_name):
    return INITIAL_PARAMETERS.get(param_name, None)