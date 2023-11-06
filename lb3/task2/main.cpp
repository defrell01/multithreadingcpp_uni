#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <iostream>
#include <chrono>

int N = 1024;

MPI_Status status;

bool isEq(int16_t* A, int16_t* B) {
    for (int i = 0; i < N * N; ++i) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}

void printMatrix(int16_t* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int16_t* generateMatrix(bool empty) {
    int16_t* matrix = new int16_t[N * N];

    for (int i = 0; i < N * N; ++i) {
        if (!empty) {
            matrix[i] = rand() % 2;
        } else {
            matrix[i] = 0;
        }
    }
    return matrix;
}

void matrixMultiplication(int16_t* matrix_a, int16_t* matrix_b, int16_t* matrix_c, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            matrix_c[i * N + j] = 0; // Initialize the result to 0
            for (int k = 0; k < N; k++) {
                matrix_c[i * N + j] += matrix_a[i * N + k] * matrix_b[k * N + j];
            }
        }
    }
}

int16_t* multiplyMatrices(int16_t* matrix1, int16_t* matrix2, int N) {
    int16_t* result = new int16_t[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                result[i * N + j] += matrix1[i * N + k] * matrix2[k * N + j];
            }
        }
    }

    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int processCount, processId, slaveTaskCount, source, dest, rows, offset;

    struct timeval start, stop;

    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    slaveTaskCount = processCount - 1;
    int finished = 0;

    int16_t* matrix_a = nullptr;
    int16_t* matrix_b = nullptr;
    int16_t* matrix_c = nullptr;
    int16_t* matrix_single = nullptr;

    if (processId == 0) {
        matrix_a = generateMatrix(false);
        matrix_b = generateMatrix(false);
        matrix_c = generateMatrix(true);
        matrix_single = generateMatrix(true);

        auto start_mult = std::chrono::high_resolution_clock::now();

        rows = N / slaveTaskCount;
        offset = 0;

        for (dest = 1; dest <= slaveTaskCount; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&matrix_a[offset * N], rows * N, MPI_INT16_T, dest, 1, MPI_COMM_WORLD);
            MPI_Send(matrix_b, N * N, MPI_INT16_T, dest, 1, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        for (int i = 1; i <= slaveTaskCount; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset * N], rows * N, MPI_INT16_T, source, 2, MPI_COMM_WORLD, &status);
        }

        
        auto stop_mult = std::chrono::high_resolution_clock::now();
        auto duration_mult = std::chrono::duration_cast<std::chrono::milliseconds>(stop_mult - start_mult).count();

        std::cout << "Время исполнения в многопотоке, мс: " << duration_mult << " \n";
        // Matrix comparison
        auto start_single = std::chrono::high_resolution_clock::now();
        matrix_single = multiplyMatrices(matrix_a, matrix_b, N);
        auto stop_single = std::chrono::high_resolution_clock::now();
        auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(stop_single - start_single).count();
        std::cout << "Время исполнения в однопотоке, мс: " << duration_single << " \n";
        std::cout << "Проверка, равны ли матрицы: " << (isEq(matrix_c, matrix_single) ? "Да" : "Нет") << '\n';

        // std::cout << "Matrix A:\n";
        // printMatrix(matrix_a, N, N);
        // std::cout << "Matrix B:\n";
        // printMatrix(matrix_b, N, N);
        // std::cout << "Matrix C (Result):\n";
        // printMatrix(matrix_c, N, N);
        // std::cout << "Matrix single (Result):\n";
        // printMatrix(matrix_single, N, N);

        delete[] matrix_a;
        delete[] matrix_b;
        delete[] matrix_c;
        delete[] matrix_single;
    }

    if (processId > 0) {
        source = 0;

        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        matrix_a = new int16_t[rows * N];
        MPI_Recv(matrix_a, rows * N, MPI_INT16_T, source, 1, MPI_COMM_WORLD, &status);
        matrix_b = new int16_t[N * N];
        MPI_Recv(matrix_b, N * N, MPI_INT16_T, source, 1, MPI_COMM_WORLD, &status);

        matrix_c = new int16_t[rows * N];
        matrixMultiplication(matrix_a, matrix_b, matrix_c, rows);

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(matrix_c, rows * N, MPI_INT16_T, 0, 2, MPI_COMM_WORLD);

        delete[] matrix_a;
        delete[] matrix_b;
        delete[] matrix_c;
    }

    MPI_Finalize();
}