#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>

void sharpen_image(cv::Mat& image, float alpha) {

    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(0, 0), 3);

    cv::addWeighted(image, 1.0 + alpha, blurred, -alpha, 0, image);
}

void rotate_scale_image(cv::Mat& image, double angle, double scale) { //pokrutit_povertet

    cv::Point2f image_center(image.cols / 2.0F, image.rows / 2.0F);

    // покрутить повертеть (с) Волков П.Л.

    cv::Mat rotationMatrix_image = cv::getRotationMatrix2D(image_center, angle, scale);

    cv::Mat image_rotated;

    cv::warpAffine(image, image_rotated, rotationMatrix_image, image.size());

    image = image_rotated;

}

int main() {
    setlocale(LC_ALL, "Russian");

    // Загрузка основного изображения
    cv::Mat big_image = cv::imread("main.png", cv::IMREAD_GRAYSCALE);

    if (big_image.empty()) {
        std::cout << "Ошибка загрузки основного изображения" << std::endl;
        return -1;
    }

    std::cout << "Основное изображение загружено! Размер: " << big_image.size() << std::endl;

    sharpen_image(big_image, 2.5f);

    double angle = 3 * 16;
    double scale = 1 + 0.05 * 16;

    cv::Mat big_rotated_scaled = big_image;



    rotate_scale_image(big_rotated_scaled, angle, scale);

    cv::Mat original, rotated;
    cv::resize(big_image, original, cv::Size(400, 400));
    cv::resize(big_rotated_scaled, rotated, cv::Size(400, 400));
    cv::imshow("Original", original);
    cv::imshow("Rotated and scaled", rotated);

    cv::Mat polar_big_image, polar_big_rotated_scaled, merged_image, result;
    int polarflags = cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG;

    int abc = (log(std::min(big_image.cols / 2, big_image.rows / 2)) * 50);

    cv::Size polarsize(abc, 360); //cv::Size(log(std::min(big_rotated_scaled.cols / 2, big_rotated_scaled.rows / 2)) * 50, 360)

    cv::Point2f big_image_center(big_image.cols / 2.0F, big_image.rows / 2.0F);
    cv::Point2f big_rotated_scaled_center(big_rotated_scaled.cols / 2.0F, big_rotated_scaled.rows / 2.0F);

    cv::warpPolar(big_image, polar_big_image, polarsize, big_image_center, std::min(big_image_center.x, big_image_center.y), polarflags);
    cv::imshow("polar big image", polar_big_image);

    cv::warpPolar(big_rotated_scaled, polar_big_rotated_scaled, polarsize, big_rotated_scaled_center, std::min(big_rotated_scaled_center.x, big_rotated_scaled_center.y), polarflags);
    cv::imshow("polar big rotated", polar_big_rotated_scaled);

    cv::vconcat(polar_big_rotated_scaled, polar_big_rotated_scaled, merged_image);
    //cv::imshow("vconcat", merged_image);
    cv::hconcat(merged_image, merged_image, merged_image);
    cv::imshow("hconcat", merged_image);
    //cv::vconcat(merged_image, merged_image, merged_image);
    //cv::hconcat(merged_image, merged_image, merged_image);
    //cv::imshow("abc", merged_image);

    //cv::resize(polar_big_rotated_scaled, polar_big_rotated_scaled, cv::Size(800, 800));

    auto start = std::chrono::high_resolution_clock::now();

    cv::matchTemplate(merged_image, polar_big_image, result, cv::TM_CCOEFF_NORMED);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Время вычисления корреляции: " << duration.count() << " секунд" << std::endl;

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);
    std::cout << "Max correlation: " << max_val << " at " << max_loc.x << ", " << max_loc.y << std::endl;

    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::resize(result, result, cv::Size(800, 800));
    cv::imshow("result", result);

    std::cout << "Detected angle: " << 360 - max_loc.y << std::endl;
    std::cout << "Detected scale: " << exp(max_loc.x / 50.0) << std::endl;

    cv::waitKey(0);

    return 0;
}
