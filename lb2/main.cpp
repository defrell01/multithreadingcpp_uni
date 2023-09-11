#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

using namespace cv;

int main()
{
    // Загрузка изображения
    Mat image = imread("input.jpg");

    if (image.empty())
    {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Разделение изображения на каналы BGR
    Mat bgr_channels[3];
    split(image, bgr_channels);

    // Создание полутоновых изображений для модифицированного синего и желтого каналов
    Mat modified_blue = Mat::zeros(image.size(), CV_8UC1);
    Mat modified_yellow = Mat::zeros(image.size(), CV_8UC1);

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

    // Сохранение полутоновых изображений
    imwrite("modified_blue.jpg", modified_blue);
    imwrite("modified_yellow.jpg", modified_yellow);

    return 0;
}
