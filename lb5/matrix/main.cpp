#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>

#define BLOCK_SIZE 32

const char* kernelSource = R"kernel(
__kernel void matrixMult(__global const short* A, __global const short* B, __global short* result, const int size) {
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    short sum = 0;
    if (row < size && col < size) {
        for (int k = 0; k < size; ++k) {
            sum += A[row * size + k] * B[k * size + col];
        }
        result[row * size + col] = sum;
    }
}
)kernel";

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();

    cl::Context context(device);

    cl::Program program(context, kernelSource);
    program.build("-cl-std=CL1.2");

    cl::CommandQueue queue(context, device);

    int size = 1024;
    size_t byte_size = size * size * sizeof(short);

    std::vector<short> h_A(size * size);
    std::vector<short> h_B(size * size);
    std::vector<short> h_C(size * size);

    // Initialize matrices
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 99);
    for (int i = 0; i < size * size; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
    }

    cl::Buffer cl_A(context, CL_MEM_READ_ONLY, byte_size);
    cl::Buffer cl_B(context, CL_MEM_READ_ONLY, byte_size);
    cl::Buffer cl_C(context, CL_MEM_WRITE_ONLY, byte_size);

    queue.enqueueWriteBuffer(cl_A, CL_TRUE, 0, byte_size, h_A.data());
    queue.enqueueWriteBuffer(cl_B, CL_TRUE, 0, byte_size, h_B.data());

    cl::Kernel kernel(program, "matrixMult");
    kernel.setArg(0, cl_A);
    kernel.setArg(1, cl_B);
    kernel.setArg(2, cl_C);
    kernel.setArg(3, size);

    cl::NDRange global(size, size);
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    queue.enqueueReadBuffer(cl_C, CL_TRUE, 0, byte_size, h_C.data());

    // Можно добавить код для проверки корректности результата

    return 0;
}
