# FFT-IFFT-Processor 项目文档

## 概述
本项目实现了一个高效的FFT(快速傅里叶变换)/IFFT(逆快速傅里叶变换)处理器，用于实时信号处理场景。

## 主要功能

- **读取灰度图像**

- **FFT运算，生成频域复数矩阵并可视化**

- **IFFT，重建原始图像**

- **低通滤波**

## 目录结构

```bash
.
├── data
│   ├── input_image.png
│   └── miku.png
├── src
│   └── fft_image_processor.cpp
├── bin
│   └── fft_processor
├── Makefile
└── output
    ├── fft_analysis_report.txt
    ├── filtered_5.png
    ├── filtered_10.png
    ├── filtered_20.png
    ├── filtered_30.png
    ├── frequency_spectrum.png
    ├── input_image.png
    └── reconstructed_image.png
