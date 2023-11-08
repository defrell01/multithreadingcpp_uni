cmake_minimum_required(VERSION 3.10)

project(MPI_MATRIX)

find_package(MPI REQUIRED)

add_executable(matrix main.cpp)

target_link_libraries(matrix MPI::MPI_CXX)