cmake_minimum_required(VERSION 3.0)
project(LB2OPENMP)

find_package(OpenCV REQUIRED)

set(SOURCES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
