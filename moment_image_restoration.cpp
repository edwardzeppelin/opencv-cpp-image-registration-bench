#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>

using namespace cv;
using namespace std;

int global_cnt = 0;

cv::Point find_image_gravity_centre(cv::Mat& img) {
    int64_t sumXLum = 0;
    int64_t sumYLum = 0;
    int64_t sumLum = 0;
    int16_t lum = 0;

    for (int16_t i = 0; i < img.rows; i++) {
        for (int16_t j = 0; j < img.cols; j++) {
            lum = static_cast<int16_t>(img.at<uchar>(i, j));
            if (lum) {
                sumLum += lum;
                sumXLum += lum * j;
                sumYLum += lum * i;
            }
        }
    }

    cv::Point centre(sumXLum / sumLum, sumYLum / sumLum);

    return centre;
}

void restore_image(cv::Mat& img) {

    cv::Point centre_figure = find_image_gravity_centre(img);

    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    int16_t lum = 0;

    for (int16_t i = 0; i < img.rows; i++) {
        for (int16_t j = 0; j < img.cols; j++) {
            lum = static_cast<int16_t>(img.at<uchar>(i, j));
            if (lum) {
                int16_t delX = j - centre_figure.x;
                int16_t delY = i - centre_figure.y;
                b += lum * ((delX * delX) - (delY * delY));
                c += lum * 2 * delX * delY;
                d += lum * ((delX * delX) + (delY * delY));
            }
        }
    }

    double compress_amount, compress_direction;

    compress_amount = sqrt((d - sqrt((c * c) + (b * b))) / (d + sqrt((c * c) + (b * b))));

    compress_direction = 0.5 * atan(c / b);

    cv::Mat rotate_minus = cv::getRotationMatrix2D(centre_figure, -compress_direction * 180 / CV_PI, 1.0);
    cv::Mat rotate_plus = cv::getRotationMatrix2D(centre_figure, compress_direction * 180 / CV_PI, 1.0);

    cv::Mat reform = cv::getRotationMatrix2D(centre_figure, 0.0, 1 / compress_amount);

    if (atan(c / b) < 1) {
        if (b > c) {
            reform.at<double>(0, 0) = 1;
            reform.at<double>(0, 2) = 1;
        }
        else
        {
            reform.at<double>(1, 1) = 1;
            reform.at<double>(1, 2) = 1;
        }
    }
    else
    {
        if (b > c) {
            reform.at<double>(1, 1) = 1;
            reform.at<double>(1, 2) = 1;
        }
        else
        {
            reform.at<double>(0, 0) = 1;
            reform.at<double>(0, 2) = 1;
        }
    }

    cv::Mat restored_img;
    cv::warpAffine(img, restored_img, rotate_plus, img.size());
    cv::warpAffine(restored_img, restored_img, reform, restored_img.size());
    cv::warpAffine(restored_img, restored_img, rotate_minus, restored_img.size());

    double M = 0.0;
    double numerator = 0.0;
    double denominator = 0.0;

    centre_figure = find_image_gravity_centre(img);

    for (int16_t i = 0; i < img.rows; i++) {
        for (int16_t j = 0; j < img.cols; j++) {
            lum = static_cast<int16_t>(img.at<uchar>(i, j));
            if (lum) {
                int16_t delX = j - centre_figure.x;
                int16_t delY = i - centre_figure.y;
                numerator += lum * sqrt((delX * delX) + (delY * delY));
                denominator += lum;

            }
        }
    }

    M = numerator / (10 * denominator);

    reform = cv::getRotationMatrix2D(centre_figure, 0.0, 1 / M);
    cv::warpAffine(restored_img, restored_img, reform, restored_img.size());

    string count = to_string(global_cnt);

    string line = "Restored in func " + count;

    cv::imshow(line, restored_img);

    global_cnt++;

}

int main()
{

    Mat img1 = imread("image_1_16.png", IMREAD_GRAYSCALE);
    cv::imshow("original 1", img1);
    Mat img2 = imread("image_2_16.png", IMREAD_GRAYSCALE);
    cv::imshow("original 2", img2);
    Mat img3 = imread("image_3_16.png", IMREAD_GRAYSCALE);
    cv::imshow("original 3", img3);

    if (img1.empty() || img2.empty() || img3.empty()) {
        cout << "Error open images" << endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    restore_image(img1);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time of calc: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    restore_image(img2);

    end = std::chrono::high_resolution_clock::now();

    duration = end - start;
    std::cout << "Time of calc: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    restore_image(img3);

    end = std::chrono::high_resolution_clock::now();

    duration = end - start;
    std::cout << "Time of calc: " << duration.count() << " seconds" << std::endl;

    waitKey(0);

    return 0;
}

