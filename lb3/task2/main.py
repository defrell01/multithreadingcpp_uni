from mpi4py import MPI
import numpy as np
import time

# Function to perform matrix multiplication in a single thread
def matrix_multiply_single_thread(matrix_a, matrix_b):
    return np.dot(matrix_a, matrix_b)

# Function to perform matrix multiplication in parallel using Open MPI
def matrix_multiply_parallel(matrix_a, matrix_b, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Broadcast matrices to all processes
    matrix_a = comm.bcast(matrix_a, root=0)
    matrix_b = comm.bcast(matrix_b, root=0)

    # Divide work among processes
    local_rows = len(matrix_a) // size
    local_matrix_a = matrix_a[rank * local_rows: (rank + 1) * local_rows, :]
    
    # Gather the results
    result = comm.gather(matrix_multiply_single_thread(local_matrix_a, matrix_b), root=0)

    if rank == 0:
        # Concatenate the results to get the final result
        result = np.concatenate(result, axis=0)
        return result

# Function to check if two matrices are equal
def matrices_equal(matrix_a, matrix_b):
    return np.array_equal(matrix_a, matrix_b)

# Function to generate random matrices
def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Set matrix dimensions
    rows = 4096
    cols = 4096

    if rank == 0:
        # Generate random matrices for multiplication
        matrix_a = generate_random_matrix(rows, cols)
        matrix_b = generate_random_matrix(cols, rows)

        # Perform matrix multiplication in a single thread
        start_time_single_thread = time.time()
        result_single_thread = matrix_multiply_single_thread(matrix_a, matrix_b)
        end_time_single_thread = time.time()
        print("Single Thread Execution Time:", end_time_single_thread - start_time_single_thread)

        # Perform matrix multiplication in parallel using Open MPI
        start_time_parallel = time.time()
        result_parallel = matrix_multiply_parallel(matrix_a, matrix_b, comm)
        end_time_parallel = time.time()
        print("Parallel Execution Time:", end_time_parallel - start_time_parallel)

        # Check if the results are equal
        if matrices_equal(result_single_thread, result_parallel):
            print("Parallel multiplication is correct.")
        else:
            print("Parallel multiplication is incorrect.")
    else:
        matrix_multiply_parallel(None, None, comm)