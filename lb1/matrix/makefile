all: build run

build:
	g++ -std=c++11 -mavx2 -mfma -mavx512f -o ../build/matrixsimd main.cpp

run:
	../build/matrixsimd