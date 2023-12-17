#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <iostream>

int mainLoop(cv::Mat& image, std::string& res, bool save)
{
    auto start = std::chrono::high_resolution_clock::now();
    // Разделение изображения на каналы BGR
    cv::Mat bgr_channels[3];
    cv::split(image, bgr_channels);

    // Создание полутоновых изображений для модифицированного синего и желтого каналов
    cv::Mat modified_blue = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat modified_yellow = cv::Mat::zeros(image.size(), CV_8UC1);

    // omp_set_nested(1);    
    // omp_set_num_threads(6);
    // #pragma omp parallel for
    for (int i = 0; i < image.rows; i++)
    {   
        
        for (int j = 0; j < image.cols; j++)
        {
            int blue = bgr_channels[0].at<uchar>(i, j);
            int green = bgr_channels[1].at<uchar>(i, j);
            int red = bgr_channels[2].at<uchar>(i, j);

            // Вычисление модифицированных значений
            int B_v = blue - (green + blue) / 2;
            int Y_v = red + green - 2 * abs(red - green) + blue;

            // Запись значений в полутоновые изображения
            modified_blue.at<uchar>(i, j) = static_cast<uchar>(B_v);
            modified_yellow.at<uchar>(i, j) = static_cast<uchar>(Y_v);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);

    if (save)
    {
        cv::imwrite("../res/" + res + "_blue_channel.jpg", modified_blue);
        cv::imwrite("../res/" + res + "_yellow_channel.jpg", modified_yellow);
    }
    

    return duration.count();
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
        std::cout << "Picture 16k loop " << i + 1 << " duration: " << duration << "ms \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    auto stop = std::chrono::steady_clock::now();

    std::cout << "16k pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";
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
        std::cout << "Picture 1 loop " << i + 1 << " duration: " << duration << " ms\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "1st pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

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
        std::cout << "Picture 1 loop " << i + 1 << " duration: " << duration << " ms \n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "2nd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

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
        std::cout << "Picture 3 loop " << i + 1 << " duration: " << duration << " ms\n";

        if (i == 0)
        {
            save = 0;
        }
    }

    stop = std::chrono::steady_clock::now();

    std::cout << "3rd pic 10 times duration: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";
}
