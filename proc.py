import numpy as np
import multiprocessing

# Define helper functions
def key_generation():
    # Generate secret key
    V = np.random.rand(2, 2)
    C = np.random.rand(2, 2)
    return {'V': V, 'C': C}

def input_encryption(W, Y, B, key):
    # Blinding
    H = np.identity(2) - 2 * np.outer(key['V'], key['V'])
    W_blind = np.dot(H, W)
    Y_blind = Y + np.dot(W_blind.T, key['C'])
    B_blind = np.dot(H, B) + key['C']
    return {'W_blind': W_blind, 'Y_blind': Y_blind, 'B_blind': B_blind}

def problem_solving(W_blind, Y_blind, B_blind):
    # Problem solving
    p = B_blind.shape[1]  # number of blocks
    q = Y_blind.shape[1]  # number of columns in Y_blind
    m = W_blind.shape[1]  # number of columns in W_blind
    
    Res = []
    for j in range(p):
        Z_j = np.dot(W_blind.T, B_blind[:, j])
        Res_j = []
        for i in range(q):
            for t in range(m):
                di_t = np.linalg.norm(Z_j[i] - Y_blind[t])**2
                if di_t < delta:
                    Res_j.append((i, t))
        Res.append(Res_j)
    return Res

def result_verification(Res, W_blind, Y_blind, B_blind, key):
    # Result verification
    valid_result = []
    for j, Res_j in enumerate(Res):
        if Res_j:
            for i, t in Res_j:
                di_t = np.linalg.norm(np.dot(W_blind.T, B_blind[:, i]) - Y_blind[t])**2
                if di_t < delta:
                    valid_result.append((i, t))
                else:
                    return "ERROR"
    return valid_result

def edge_server_process(enc_data, results_queue):
    W_blind, Y_blind, B_blind = enc_data['W_blind'], enc_data['Y_blind'], enc_data['B_blind']
    Res_j = problem_solving(W_blind, Y_blind, B_blind)
    results_queue.put(Res_j)

# Define parameters
lambda_value = 0.5
n = 2
W = np.random.rand(n, 2)
Y = np.random.rand(2, 2)
B = np.random.rand(n, 2)
delta = 0.1
p = 3  # Number of edge servers

# Protocol execution
key = key_generation()
enc_data = input_encryption(W, Y, B, key)

# Create a queue for storing results from edge servers
results_queue = multiprocessing.Queue()

# Create processes for each edge server
processes = []
for _ in range(p):
    process = multiprocessing.Process(target=edge_server_process, args=(enc_data, results_queue))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

# Retrieve results from the queue
results_from_edge_servers = [results_queue.get() for _ in range(p)]

# User side: Receive calculation results from edge servers and verify
valid_results = result_verification(results_from_edge_servers, enc_data['W_blind'], enc_data['Y_blind'], enc_data['B_blind'], key)

print("Valid result:", valid_results)
