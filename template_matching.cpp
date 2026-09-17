#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

void sharpen_image(cv::Mat& image, float alpha) {
    
    cv::Mat blurred;

    cv::GaussianBlur(image, blurred, cv::Size(0, 0), 3); // Размер ядра (0,0) - вычисляется из сигмы=3

    cv::addWeighted(image, 1.0 + alpha, blurred, -alpha, 0, image);
}

void rotate_scale_image(cv::Mat& image, double angle, double scale) {

    cv::Point2f image_center(image.cols / 2.0F, image.rows / 2.0F);

    // покрутить повертеть (с) Волков П.Л.

    cv::Mat rotationMatrix_image = cv::getRotationMatrix2D(image_center, angle, scale);

    cv::Mat image_rotated;

    cv::warpAffine(image, image_rotated, rotationMatrix_image, image.size());

    image = image_rotated;

}

int main() {

    setlocale(LC_ALL, "Russian");

    // Загрузка изображений
    cv::Mat big_image = cv::imread("main.png", cv::IMREAD_GRAYSCALE);
    cv::Mat template1 = cv::imread("own_16.png", cv::IMREAD_GRAYSCALE);
    cv::Mat template2 = cv::imread("foreign_16.png", cv::IMREAD_GRAYSCALE);

    // Проверка загрузки
    if (big_image.empty()) {
        std::cout << "Ошибка загрузки большого изображения: " << std::endl;
        return -1;
    }
    if (template1.empty()) {
        std::cout << "Ошибка загрузки эталона 1: " << std::endl;
        return -1;
    }
    if (template2.empty()) {
        std::cout << "Ошибка загрузки эталона 2: " << std::endl;
        return -1;
    }

    std::cout << "Изображения успешно загружены!" << std::endl;

    sharpen_image(big_image, 2.5f);
    sharpen_image(template1, 2.5f);
    sharpen_image(template2, 2.5f);

    rotate_scale_image(big_image, 10.0, 1.1); //0,9, 0,925, 0,95, 0,975, 1,0, 1,025, 1,05, 1,075, 1,1
    rotate_scale_image(template1, 10.0, 1.1);
    rotate_scale_image(template2, 10.0, 1.1);

    // -------------------------------------- Correlation -----------------------------------------
    cv::Mat correlation1, correlation2;

    double worktime1 = 0;
    double worktime2 = 0;

    auto start1 = std::chrono::high_resolution_clock::now();
    
    cv::matchTemplate(big_image, template1, correlation1, cv::TM_CCOEFF_NORMED);

    auto end1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration1 = end1 - start1;

    worktime1 = duration1.count();

    auto start2 = std::chrono::high_resolution_clock::now();

    cv::matchTemplate(big_image, template2, correlation2, cv::TM_CCOEFF_NORMED);

    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration2 = end2 - start2;

    worktime2 = duration2.count();

    double max_val1, max_val2;
    cv::Point max_loc1, max_loc2;

    cv::minMaxLoc(correlation1, nullptr, &max_val1, nullptr, &max_loc1);
    cv::minMaxLoc(correlation2, nullptr, &max_val2, nullptr, &max_loc2);

    std::cout << "Максимальная корреляция с эталоном 1: " << max_val1 << "(" << max_loc1.x << ", " << max_loc1.y << ")" << std::endl;
    std::cout << "Максимальная корреляция с эталоном 2: " << max_val2 << "(" << max_loc2.x << ", " << max_loc2.y << ")" << std::endl;

    // создание изображения с результатами
    cv::Mat result_image;
    cv::cvtColor(big_image, result_image, cv::COLOR_GRAY2BGR);

    double threshold = 0.7; // порог обнаружения
    bool found1 = (max_val1 > threshold);
    bool found2 = (max_val2 > threshold);

    // Определение и отображение результатов
    if (found1 || found2) {
        if (found1 && (!found2 || max_val1 > max_val2)) {
            std::cout << "Обнаружен эталон 1" << std::endl;
            std::cout << "Позиция: x=" << max_loc1.x << ", y=" << max_loc1.y << std::endl;

            // Рисуем прямоугольник вокруг найденного объекта
            cv::rectangle(result_image, cv::Rect(max_loc1.x, max_loc1.y, template1.cols, template1.rows), cv::Scalar(0, 255, 0), 2);
            cv::putText(result_image, "It's here, for sure", cv::Point(max_loc1.x, max_loc1.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        else if (found2 && (!found1 || max_val2 > max_val1)) {
            std::cout << "Обнаружен эталон 2" << std::endl;
            std::cout << "Позиция: x=" << max_loc2.x << ", y=" << max_loc2.y << std::endl;

            // Рисуем прямоугольник вокруг найденного объекта
            cv::rectangle(result_image, cv::Rect(max_loc2.x, max_loc2.y, template2.cols, template2.rows), cv::Scalar(0, 255, 0), 2);
            cv::putText(result_image, "It's here, for sure", cv::Point(max_loc2.x, max_loc2.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        else {
            std::cout << "Обнаружены оба эталона" << std::endl;
            std::cout << "Позиция: x=" << max_loc2.x << ", y=" << max_loc2.y << std::endl;

            cv::rectangle(result_image, cv::Rect(max_loc2.x, max_loc2.y, template2.cols, template2.rows), cv::Scalar(0, 0, 255), 2);
            cv::putText(result_image, "Two of them here!", cv::Point(max_loc2.x, max_loc2.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
    }
    else {
        std::cout << "Эталоны не обнаружены" << std::endl;
    }

    std::cout << "Время расчёта корреляции эталона 1: " << worktime1 << std::endl;
    std::cout << "Время расчёта корреляции эталона 2: " << worktime2 << std::endl;

    //------------------------------ Show everything ------------------------------------

    cv::Mat display_corr1, display_corr2;

    cv::normalize(correlation1, display_corr1, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(correlation2, display_corr2, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat resized_corr1, resized_corr2;

    cv::resize(display_corr1, resized_corr1, cv::Size(600, 600));
    cv::resize(display_corr2, resized_corr2, cv::Size(600, 600));

    cv::Mat resized_big_image, resized_result_image;

    cv::resize(big_image, resized_big_image, cv::Size(700, 700));
    cv::resize(result_image, resized_result_image, cv::Size(700, 700));

    cv::imshow("Main image", resized_big_image);
    cv::imshow("Template 1", template1);
    cv::imshow("Template 2", template2);
    cv::imshow("Result", resized_result_image);
    cv::imshow("Corr field Temp 1", resized_corr1);
    cv::imshow("Corr field Temp 2", resized_corr2);

    cv::waitKey(0);

    return 0;
}