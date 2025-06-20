# FFT-IFFT-Processor 项目文档

## 概述
本项目实现了一个高效的FFT(快速傅里叶变换)/IFFT(逆快速傅里叶变换)处理器，用于实时信号处理场景。

## 主要功能

- **灰度图像处理**：读取并处理灰度图像
- **FFT运算**：执行傅里叶变换生成频域复数矩阵
- **频谱可视化**：可视化频率谱分析结果
- **IFFT重建**：执行逆变换重建原始图像
- **低通滤波**：实现多级滤波能力

## 使用方法
### Ubuntu/Debian系统
```bash
sudo apt install libopencv-dev  # 安装OpenCV依赖库
make                            # 编译项目
./fft_processor                 # 运行处理器程序

### Ubuntu/Debian
```bash
sudo apt install libopencv-dev
make
./fft_processor
