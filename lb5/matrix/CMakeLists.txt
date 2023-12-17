cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Нахождение OpenCL
find_package(OpenCL REQUIRED)

# Если вы используете OpenCV, раскомментируйте эти строки
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# Добавление исполняемого файла
add_executable(MatrixMultiplication main.cpp)

# Линковка с OpenCL
target_link_libraries(MatrixMultiplication OpenCL::OpenCL)

# Линковка с OpenCV, если он используется
# target_link_libraries(MatrixMultiplication ${OpenCV_LIBS})
