#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <iostream>
#include <chrono>
#include <string>

__global__ void modifyChannels(const uchar* src, uchar* blueChannel, uchar* yellowChannel, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        uchar red = src[idx * 3 + 2];
        uchar green = src[idx * 3 + 1];
        uchar blue = src[idx * 3];

        // Ваши формулы для модификации каналов:
        blueChannel[idx] = blue - (green + blue) / 2;
        yellowChannel[idx] = red + green - 2 * (abs(red - green) + blue);
    }
}

int64_t mainLoop(cv::Mat& src, std::string &res, uint save)
{

    auto start = std::chrono::steady_clock::now();

    // Подготавливаем GPU матрицы
    cv::cuda::GpuMat gpuSrc, gpuBlueChannel, gpuYellowChannel;
    gpuSrc.upload(src);
    gpuBlueChannel.create(src.size(), CV_8UC1);
    gpuYellowChannel.create(src.size(), CV_8UC1);

    // Вычисляем размеры сетки и блока
    const dim3 block(16, 16);
    const dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

    // Запускаем ядро CUDA
    modifyChannels<<<grid, block>>>(gpuSrc.data, gpuBlueChannel.data, gpuYellowChannel.data, src.rows, src.cols);
    cudaDeviceSynchronize();

    // Скачиваем результаты
    cv::Mat blueChannel, yellowChannel;
    gpuBlueChannel.download(blueChannel);
    gpuYellowChannel.download(yellowChannel);
    
    if (save)
    {
        cv::imwrite("../res/" + res + "_blue_channel.jpg", blueChannel);
        cv::imwrite("../res/" + res + "_yellow_channel.jpg", yellowChannel);
    }
    
    auto stop = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();

    return duration;
    
}

int main() 
{
    std::string input = "../img/16k.jpg";
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
        std::cout << "Picture 16k loop " << i+1 << " duration: " << duration << "mcs \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    auto stop = std::chrono::steady_clock::now();

    std::cout << "16k pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << " mcs\n";
    std::cout << "\n";

    input = "../img/1.jpg";
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
        std::cout << "Picture 1 loop " << i+1 << " duration: " << duration << " mcs\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "1st pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << " mcs\n";

    std::cout << "\n";

    input = "../img/2.jpg";
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
        std::cout << "Picture 1 loop " << i+1 << " duration: " << duration << " mcs \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "2nd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << " mcs\n";
    
    std::cout << "\n";

    input = "../img/3.jpg";
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
        std::cout << "Picture 3 loop " << i+1 << " duration: " << duration << " mcs\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "3rd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << " mcs\n";

    

    
}
