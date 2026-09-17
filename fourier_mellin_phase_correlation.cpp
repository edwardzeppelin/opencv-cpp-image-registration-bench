#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>

using namespace std;
using namespace cv;


void rotate_scale_shift_image(cv::Mat& image, double angle, double scale, double xshift, double yshift) { //pokrutit_povertet

	cv::Point2f image_center(image.cols / 2.0F, image.rows / 2.0F);

	// покрутить повертеть (с) Волков П.Л.

	cv::Mat image_rotated;

	cv::Mat rotationMatrix_image = cv::getRotationMatrix2D(image_center, angle, scale);

	cv::warpAffine(image, image_rotated, rotationMatrix_image, image.size());

	cv::Mat shiftMatrix = (Mat_<double>(2, 3) << 1, 0, xshift * image_rotated.cols, 0, 1, yshift * image_rotated.rows);

	cv::warpAffine(image_rotated, image_rotated, shiftMatrix, image_rotated.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	image = image_rotated;

}

// Convert image to float and compute log-magnitude of its DFT
static void dftLogMagnitude(const Mat& src8u, Mat& magOut)
{
	Mat src;
	if (src8u.type() != CV_32F)
		src8u.convertTo(src, CV_32F);
	else
		src = src8u;

	// make complex image
	Mat planes[] = { src, Mat::zeros(src.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	// perform DFT
	dft(complexI, complexI);

	// compute magnitude
	split(complexI, planes);
	magnitude(planes[0], planes[1], magOut);

	// log scale for visualization/stability (add 1 to avoid log(0))
	magOut += Scalar::all(1.0);
	log(magOut, magOut);

	// shift the quadrants so that the DC component is at the center
	// (not strictly necessary for phase correlation in log-polar but helpful)
	int cx = magOut.cols / 2;
	int cy = magOut.rows / 2;
	Mat q0(magOut, Rect(0, 0, cx, cy));   // Top-Left
	Mat q1(magOut, Rect(cx, 0, magOut.cols - cx, cy));  // Top-Right
	Mat q2(magOut, Rect(0, cy, cx, magOut.rows - cy));  // Bottom-Left
	Mat q3(magOut, Rect(cx, cy, magOut.cols - cx, magOut.rows - cy)); // Bottom-Right

	Mat tmp;
	q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
	q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

int main()
{
	// --- Load ---
	Mat img = imread("original_crop.png", IMREAD_GRAYSCALE);
	Mat tpl = imread("original_crop1.png", IMREAD_GRAYSCALE);
	if (img.empty() || tpl.empty()) {
		cerr << "Can't load images. Put original.png and original_crop.png next to the exe." << endl;
		return -1;
	}

	rotate_scale_shift_image(img, 0.0, 1.0, 0.0, 0.0);
	rotate_scale_shift_image(tpl, 0.0, 1.0, 0.0, 0.0);

	// --- Optional: show originals ---
	namedWindow("img", WINDOW_AUTOSIZE);
	namedWindow("tpl", WINDOW_AUTOSIZE);
	imshow("img", img);
	imshow("tpl", tpl);

	// --- Pad template to image size (centered) ---
	Mat tpl32;
	tpl.convertTo(tpl32, CV_32F);

	Mat tplPadded = Mat::zeros(img.size(), CV_32F);
	int x0 = (img.cols - tpl.cols) / 2;
	int y0 = (img.rows - tpl.rows) / 2;
	if (x0 < 0 || y0 < 0) {
		cerr << "Template is larger than image!" << endl;
		return -1;
	}
	tpl32.copyTo(tplPadded(Rect(x0, y0, tpl.cols, tpl.rows)));

	// --- Compute log-magnitude of DFT for both images ---
	Mat magImg, magTpl;
	dftLogMagnitude(img, magImg);
	dftLogMagnitude(tplPadded, magTpl);

	cv::Mat dftLogNorm, dftLog8U;
	cv::normalize(magImg, dftLogNorm, 0, 255, cv::NORM_MINMAX);
	dftLogNorm.convertTo(dftLog8U, CV_8U);
	cv::imshow("DFT Log Magnitude", dftLog8U);

	cv::Mat dftLogNorm2, dftLog8U2;
	cv::normalize(magTpl, dftLogNorm2, 0, 255, cv::NORM_MINMAX);
	dftLogNorm2.convertTo(dftLog8U2, CV_8U);
	cv::imshow("DFT Log Magnitude2", dftLog8U2);

	// --- Log-polar transform of magnitude spectra ---
	Point2f center((float)magImg.cols / 2.0f, (float)magImg.rows / 2.0f);
	// M parameter controls sampling in log-polar; choose proportional to rows
	double M = 80.0; // you can tune this (20..80)
	Mat lpImg, lpTpl;
	logPolar(magImg, lpImg, center, M, cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG); //warpfilloutliers
	logPolar(magTpl, lpTpl, center, M, cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS + cv::WARP_POLAR_LOG);

	cv::Mat lpImgNorm, lpTplNorm;
	cv::Mat lpImg8U, lpTpl8U;

	cv::normalize(lpImg, lpImgNorm, 0, 255, cv::NORM_MINMAX);
	cv::normalize(lpTpl, lpTplNorm, 0, 255, cv::NORM_MINMAX);

	lpImgNorm.convertTo(lpImg8U, CV_8U);
	lpTplNorm.convertTo(lpTpl8U, CV_8U);

	cv::imshow("LogPolar Image", lpImg8U);
	cv::imshow("LogPolar Template", lpTpl8U);

	// --- Apply a Hanning window to reduce edge effects (recommended) ---
	Mat hann;
	createHanningWindow(hann, lpImg.size(), CV_32F);
	Mat lpImgWin = lpImg.mul(hann);
	Mat lpTplWin = lpTpl.mul(hann);

	// --- Phase correlation in log-polar domain: gives rotation & scale ---
	double corrScaleRot = 0.0;
	Point2d shiftLogPolar = phaseCorrelate(lpImgWin, lpTplWin, Mat(), &corrScaleRot);

	// Interpret shift: shift.x -> log-radius (scale), shift.y -> angle
	double angle = -shiftLogPolar.y * 360.0 / lpTpl.rows; // degrees
	double scale = exp(shiftLogPolar.x / M);

	cout << "Found scale = " << scale << " , angle = " << angle << " deg" << endl;
	cout << "Log-polar phaseCorrelate response = " << corrScaleRot << endl;


	/*
	// --- Warp (rotate+scale) the PADDED template to align with image ---
	// Use center of image as rotation center
	Mat rot = getRotationMatrix2D(center, angle, scale);
	Mat tplWarped;
	warpAffine(tplPadded, tplWarped, rot, tplPadded.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	// --- Now find translation using phaseCorrelate in spatial domain ---
	Mat imgF, tplWarpedF;
	img.convertTo(imgF, CV_32F);
	tplWarped.convertTo(tplWarpedF, CV_32F);

	// Apply Hanning window in spatial domain as well
	Mat hann2;
	createHanningWindow(hann2, imgF.size(), CV_32F);
	Mat imgFwin = imgF.mul(hann2);
	Mat tplFwin = tplWarpedF.mul(hann2);

	double corrTranslation = 0.0;
	Point2d shift = phaseCorrelate(imgFwin, tplFwin, Mat(), &corrTranslation);

	cout << "Translation (dx,dy) = (" << shift.x << ", " << shift.y << ")" << endl;
	cout << "Spatial phaseCorrelate response = " << corrTranslation << endl;

	
	// --- Optionally: compute normalized cross-correlation (NCC) near the found location ---
	// Because we have tplWarped of same size as img (mostly zeros except where template is),
	// we can extract the bounding box where template originally was (after warp+shift) and compute NCC

	// Compute the location of the template's top-left after reverse-transform:
	// The original template was at (x0,y0) in tplPadded before warping.
	// Find that corner after warp: transform its 4 corners and estimate bounding rect.
	std::vector<Point2f> cornersIn, cornersOut;
	cornersIn.push_back(Point2f((float)x0, (float)y0));
	cornersIn.push_back(Point2f((float)(x0 + tpl.cols), (float)y0));
	cornersIn.push_back(Point2f((float)x0, (float)(y0 + tpl.rows)));
	cornersIn.push_back(Point2f((float)(x0 + tpl.cols), (float)(y0 + tpl.rows)));
	transform(cornersIn, cornersOut, rot);

	// Shift by found translation (phaseCorrelate gives how much tplWarped should be shifted to match img)
	for (size_t i = 0; i < cornersOut.size(); ++i) {
		cornersOut[i].x += (float)shift.x;
		cornersOut[i].y += (float)shift.y;
	}

	// Bounding rect
	Rect bbox = boundingRect(cornersOut);
	// Clamp
	bbox &= Rect(0, 0, img.cols, img.rows);

	double nccVal = -1.0;
	if (bbox.width > 0 && bbox.height > 0 && bbox.width >= tpl.cols / 10) {
		Mat roiImg = img(bbox);
		// Build warped template crop that corresponds to bbox
		Mat tplWarped8;
		tplWarped.convertTo(tplWarped8, CV_8U);
		Mat roiTpl = tplWarped8(bbox);

		//imshow("roiImg", roiImg);
		//imshow("roiTpl", roiTpl);

		// matchTemplate expects template smaller than image; roiTpl may be same or smaller
		Mat result;
		if (roiImg.cols >= roiTpl.cols && roiImg.rows >= roiTpl.rows) {
			matchTemplate(roiImg, roiTpl, result, TM_CCORR_NORMED);
			double maxVal;
			Point maxLoc;
			minMaxLoc(result, NULL, &maxVal, NULL, &maxLoc);
			nccVal = maxVal;
		}
	}

	if (nccVal >= 0)
		cout << "NCC (matchTemplate TM_CCORR_NORMED) near found location = " << nccVal << endl;
	else
		cout << "NCC not computed (bounding box too small or out of range)." << endl;
	*/

	waitKey();
	return 0;
}
