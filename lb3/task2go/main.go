package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/mpi2/go"
)

func matrixMultiplySingleThread(matrixA, matrixB [][]float64) [][]float64 {
	rowsA := len(matrixA)
	colsA := len(matrixA[0])
	colsB := len(matrixB[0])

	result := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		result[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += matrixA[i][k] * matrixB[k][j]
			}
		}
	}
	return result
}

func matrixMultiplyParallel(matrixA, matrixB [][]float64, rank, size int) [][]float64 {
	rowsA := len(matrixA)
	colsA := len(matrixA[0])
	colsB := len(matrixB[0])

	localRows := rowsA / size
	localMatrixA := make([][]float64, localRows)
	for i := 0; i < localRows; i++ {
		localMatrixA[i] = make([]float64, colsA)
		for j := 0; j < colsA; j++ {
			localMatrixA[i][j] = matrixA[rank*localRows+i][j]
		}
	}

	result := make([][]float64, localRows)
	for i := 0; i < localRows; i++ {
		result[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += localMatrixA[i][k] * matrixB[k][j]
			}
		}
	}
	return result
}

func generateRandomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = rand.Float64()
		}
	}
	return matrix
}

func matricesEqual(matrixA, matrixB [][]float64) bool {
	rowsA := len(matrixA)
	colsA := len(matrixA[0])
	rowsB := len(matrixB)
	colsB := len(matrixB[0])

	if rowsA != rowsB || colsA != colsB {
		return false
	}

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			if matrixA[i][j] != matrixB[i][j] {
				return false
			}
		}
	}
	return true
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Initialize MPI
	mpi.Init(nil, nil)

	rank := mpi.Rank()
	size := mpi.Size()

	rows := 1000
	cols := 1000

	if rank == 0 {
		matrixA := generateRandomMatrix(rows, cols)
		matrixB := generateRandomMatrix(cols, rows)

		startTimeSingleThread := time.Now()
		resultSingleThread := matrixMultiplySingleThread(matrixA, matrixB)
		endTimeSingleThread := time.Now()
		fmt.Println("Single Thread Execution Time:", endTimeSingleThread.Sub(startTimeSingleThread))

		startTimeParallel := time.Now()
		resultParallel := matrixMultiplyParallel(matrixA, matrixB, rank, size)
		endTimeParallel := time.Now()
		fmt.Println("Parallel Execution Time:", endTimeParallel.Sub(startTimeParallel))

		if matricesEqual(resultSingleThread, resultParallel) {
			fmt.Println("Parallel multiplication is correct.")
		} else {
			fmt.Println("Parallel multiplication is incorrect.")
		}
	} else {
		matrixMultiplyParallel(nil, nil, rank, size)
	}

	// Finalize MPI
	mpi.Finalize()
}