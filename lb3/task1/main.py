from mpi4py import MPI
import hashlib
import time

charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def single_thread_check_passwords(password_length, target_hash) -> None:
    for i in range(0, len(charset) ** password_length):
        password = ''.join(charset[(i // (len(charset) ** j)) % len(charset)] for j in range(password_length))

        if hashlib.md5(password.encode()).hexdigest() == target_hash:
            print(f"Password: {password}")
            return

def multithreading_check_passwords(rank, size, password_length, target_hash) -> None:
    combinations_count = len(charset) ** password_length

    combinations_count_per_process = combinations_count // size
    start = rank * combinations_count_per_process
    end = combinations_count if rank == size - 1 else (rank + 1) * combinations_count_per_process
    for i in range(start, end):
        password = ''.join(charset[(i // (len(charset) ** j)) % len(charset)] for j in range(password_length))

        if hashlib.md5(password.encode()).hexdigest() == target_hash:
            print(f"Password: {password}")
            return


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    process_num = comm.Get_size()

    if process_num < 2:
        raise Exception("You should run more processes")

    if rank == 0:
        password = "8Aap" # Здесь может быть ваш ввод
        password_length = len(password)
        target_hash = hashlib.md5(password.encode()).hexdigest()
    else:
        password_length = None
        target_hash = None

    password_length = comm.bcast(password_length, root=0)
    target_hash = comm.bcast(target_hash, root=0)

    comm.Barrier()  # Синхронизация начала
    start = time.time()
    multithreading_check_passwords(rank, process_num, password_length, target_hash)
    end = time.time()
    time_spent = end - start
    comm.Barrier()  # Синхронизация окончания

    total_time = comm.reduce(time_spent, op=MPI.MAX, root=0)
    
    if rank == 0:
        print(f"Total time: {total_time} seconds")

