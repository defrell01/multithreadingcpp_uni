all: build run

build:
	g++ -std=c++11 -mavx2 -mfma -mavx512f -o ../../build/threadingsimd main.cpp

run:
	../../build/threadingsimd