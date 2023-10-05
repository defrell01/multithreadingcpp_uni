#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <iostream>

int main()
{
    // Загрузка изображения
    cv::Mat image = cv::imread("../pics/input.jpg");

    if (image.empty())
    {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    // Разделение изображения на каналы BGR
    cv::Mat bgr_channels[3];
    cv::split(image, bgr_channels);

    // Создание полутоновых изображений для модифицированного синего и желтого каналов
    cv::Mat modified_blue = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat modified_yellow = cv::Mat::zeros(image.size(), CV_8UC1);

    // omp_set_nested(1);    
    // omp_set_num_threads(6);
    #pragma omp parallel for
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

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    std::cout << duration.count() << "\n";

    // Сохранение полутоновых изображений
    cv::imwrite("../pics/modified_blue.jpg", modified_blue);
    cv::imwrite("../pics/modified_yellow.jpg", modified_yellow);

    return 0;
}
