#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// OpenCL ядро для модификации каналов
const char* kernelSource = R"kernel(
__kernel void modifyChannels(__global const uchar* src, __global uchar* blueChannel, __global uchar* yellowChannel, int rows, int cols) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        uchar red = src[idx * 3 + 2];
        uchar green = src[idx * 3 + 1];
        uchar blue = src[idx * 3];

        blueChannel[idx] = blue - (green + blue) / 2;
        yellowChannel[idx] = red + green - 2 * (abs(red - green) + blue);
    }
}
)kernel";

int main() {
    // Инициализация OpenCL
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

    // Загрузка изображения с помощью OpenCV
    cv::Mat src = cv::imread("../img/1.jpg");
    if (src.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Создание буферов OpenCL
    cl::Buffer clSrc(context, CL_MEM_READ_ONLY, src.total() * src.elemSize());
    cl::Buffer clBlueChannel(context, CL_MEM_WRITE_ONLY, src.total());
    cl::Buffer clYellowChannel(context, CL_MEM_WRITE_ONLY, src.total());

    // Копирование данных в буфер
    queue.enqueueWriteBuffer(clSrc, CL_TRUE, 0, src.total() * src.elemSize(), src.data);

    // Установка аргументов и запуск ядра
    cl::Kernel kernel(program, "modifyChannels");
    kernel.setArg(0, clSrc);
    kernel.setArg(1, clBlueChannel);
    kernel.setArg(2, clYellowChannel);
    kernel.setArg(3, src.rows);
    kernel.setArg(4, src.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(src.cols, src.rows));
    queue.finish();

    // Получение результата
    cv::Mat blueChannel(src.size(), CV_8UC1);
    cv::Mat yellowChannel(src.size(), CV_8UC1);
    queue.enqueueReadBuffer(clBlueChannel, CL_TRUE, 0, src.total(), blueChannel.data);
    queue.enqueueReadBuffer(clYellowChannel, CL_TRUE, 0, src.total(), yellowChannel.data);

    // Сохранение изображений
    cv::imwrite("../res/blue_channel.jpg", blueChannel);
    cv::imwrite("../res/yellow_channel.jpg", yellowChannel);

    return 0;
}
