#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

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

int mainLoop(cv::Mat& src, std::string& res, bool save)
{
    auto start = std::chrono::high_resolution_clock::now();
    
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
    if (save)
    {
        cv::imwrite("res/" + res + "_blue_channel.jpg", blueChannel);
        cv::imwrite("res/" + res + "_yellow_channel.jpg", yellowChannel);
    }
    

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    return duration;
}

int main() 
{
    std::string input = "img/16k.jpg";
    std::string output = "16k";
    int save = 1;

    cv::Mat src = cv::imread(input);
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        int64_t duration = mainLoop(src, output, save);
        std::cout << "Picture 16k loop " << i + 1 << " duration: " << duration << "ms \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    auto stop = std::chrono::steady_clock::now();

    std::cout << "16k pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";
    std::cout << "\n";

    input = "img/1.jpg";
    output = "1";
    save = 1;

    src = cv::imread(input);
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        int64_t duration = mainLoop(src, output, save);
        std::cout << "Picture 1 loop " << i + 1 << " duration: " << duration << " ms\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "1st pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

    std::cout << "\n";

    input = "img/2.jpg";
    output = "2";
    save = 1;

    src = cv::imread(input);
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        int64_t duration = mainLoop(src, output, save);
        std::cout << "Picture 1 loop " << i + 1 << " duration: " << duration << " ms \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "2nd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

    std::cout << "\n";

    input = "img/3.jpg";
    output = "3";
    save = 1;

    src = cv::imread(input);
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < 10; ++i)
    {
        int64_t duration = mainLoop(src, output, save);
        std::cout << "Picture 3 loop " << i + 1 << " duration: " << duration << " ms\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "3rd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

}