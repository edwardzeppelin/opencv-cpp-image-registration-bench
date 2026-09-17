# OpenCV Image Alignment, Invariant Matching & Geometric Restoration

A C++ and OpenCV toolkit implementing algorithms for 2D image registration, template matching, moment-based geometric restoration, and invariant phase correlation in the frequency domain.

---

## 📌 Overview

This repository contains C++ implementations of several fundamental computer vision routines designed for pattern detection, rotation/scale estimation, and geometric image unwarping:

1. **Sharpened Template Detection**: Enhanced normalized cross-correlation matching with preliminary Gaussian unsharp masking.
2. **Log-Polar Scale & Rotation Estimation**: Spatial log-polar warping combined with correlation fields to calculate affine transformations.
3. **Moment-Based Image Unwarping**: Extraction of geometric gravity centers and second-order luminance moments to reconstruct squished or rotated binary objects.
4. **Fourier-Mellin Phase Correlation**: Frequency-domain matching using 2D Discrete Fourier Transform (DFT) log-magnitude spectra, log-polar re-sampling, Hanning windowing, and phase correlation.

---

## 📂 Project Structure

| File Name | Primary Algorithm / Technique | Key OpenCV Functions |
| :--- | :--- | :--- |
| `template_matching.cpp` | Template matching with image sharpening and affine transformation | `cv::matchTemplate`, `cv::GaussianBlur`, `cv::warpAffine` |
| `log_polar_alignment.cpp` | Spatial log-polar warping for angle & scale extraction | `cv::warpPolar`, `cv::matchTemplate`, `cv::vconcat` |
| `moment_image_restoration.cpp` | Inertial axis & centroid recovery via image intensity moments | `cv::getRotationMatrix2D`, `cv::warpAffine` |
| `fourier_mellin_phase_correlation.cpp` | Log-Polar DFT phase correlation invariant matching | `cv::dft`, `cv::logPolar`, `cv::phaseCorrelate` |

---

## 🛠 Features & Highlights

* **Image Enhancement**: Custom unsharp masking pipeline (`sharpen_image`) using Gaussian blur overlays.
* **Moment Analysis**: Intensity-weighted center of gravity calculation for binary and grayscale shapes.
* **Frequency Domain Analysis**: Spectrum quadrant shifting, log-magnitude scaling, and spectral windowing via `createHanningWindow`.
* **Sub-pixel Transformation Detection**: Precise computation of scaling factor ($e^{\Delta x / M}$) and rotation angle ($\theta$).

---

## 🚀 Prerequisites & Compilation

### Requirements
* **Compiler**: C++11 or higher (GCC, Clang, or MSVC)
* **Library**: OpenCV 4.x (`opencv_core`, `opencv_imgproc`, `opencv_highgui`, `opencv_imgcodecs`)

### Building with GCC

```bash
# Compile template matching module
g++ -O2 template_matching.cpp -o template_matching `pkg-config --cflags --libs opencv4`

# Compile Fourier-Mellin phase correlation module
g++ -O2 fourier_mellin_phase_correlation.cpp -o fourier_mellin `pkg-config --cflags --libs opencv4`
