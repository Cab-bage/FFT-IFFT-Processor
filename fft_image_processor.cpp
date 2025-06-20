#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/stat.h>  // 用于创建目录

using namespace cv;
using namespace std;

class FFTImageProcessor {
private:
    Mat originalImage;
    vector<vector<complex<double>>> frequencyDomain;
    int width, height;
    string outputDir = "output/";  // 输出目录
    
    // 位逆序置换
    void bitReverse(vector<complex<double>>& data) {
        int n = data.size();
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j) {
                swap(data[i], data[j]);
            }
        }
    }
    
    // 1D FFT实现
    void fft1D(vector<complex<double>>& data, bool inverse = false) {
        int n = data.size();
        bitReverse(data);
        
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2 * M_PI / len * (inverse ? 1 : -1);
            complex<double> wlen(cos(angle), sin(angle));
            
            for (int i = 0; i < n; i += len) {
                complex<double> w(1);
                for (int j = 0; j < len / 2; j++) {
                    complex<double> u = data[i + j];
                    complex<double> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        
        if (inverse) {
            for (auto& x : data) {
                x /= n;
            }
        }
    }
    
    // 2D FFT实现
    void fft2D(vector<vector<complex<double>>>& matrix, bool inverse = false) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        // 对每一行进行FFT
        for (int i = 0; i < rows; i++) {
            fft1D(matrix[i], inverse);
        }
        
        // 对每一列进行FFT
        for (int j = 0; j < cols; j++) {
            vector<complex<double>> column(rows);
            for (int i = 0; i < rows; i++) {
                column[i] = matrix[i][j];
            }
            fft1D(column, inverse);
            for (int i = 0; i < rows; i++) {
                matrix[i][j] = column[i];
            }
        }
    }
    
    // 将大小调整为2的幂次
    int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }
    
    // 计算MSE
    double calculateMSE(const Mat& img1, const Mat& img2) {
        Mat diff;
        absdiff(img1, img2, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);
        Scalar mse = mean(diff);
        return mse[0];
    }
    
    // 计算PSNR
    double calculatePSNR(const Mat& img1, const Mat& img2) {
        double mse = calculateMSE(img1, img2);
        if (mse == 0) return INFINITY;
        return 20 * log10(255.0 / sqrt(mse));
    }
    
public:
    // 读取灰度图像
    bool loadImage(const string& filename) {
        originalImage = imread(filename, IMREAD_GRAYSCALE);
        if (originalImage.empty()) {
            cout << "无法读取图像: " << filename << endl;
            return false;
        }
        
        cout << "原始图像大小: " << originalImage.cols << "x" << originalImage.rows << endl;
        
        // 调整图像大小为2的幂次，便于FFT计算
        width = nextPowerOf2(originalImage.cols);
        height = nextPowerOf2(originalImage.rows);
        
        Mat paddedImage;
        copyMakeBorder(originalImage, paddedImage, 0, height - originalImage.rows, 
                      0, width - originalImage.cols, BORDER_CONSTANT, Scalar(0));
        
        // 转换为复数矩阵
        frequencyDomain.resize(height, vector<complex<double>>(width));
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (i < paddedImage.rows && j < paddedImage.cols) {
                    frequencyDomain[i][j] = complex<double>(paddedImage.at<uchar>(i, j), 0);
                } else {
                    frequencyDomain[i][j] = complex<double>(0, 0);
                }
            }
        }
        
        cout << "填充后图像大小: " << width << "x" << height << endl;
        return true;
    }
    
    // 执行FFT
    void performFFT() {
        cout << "正在执行FFT..." << endl;
        fft2D(frequencyDomain, false);
        cout << "FFT完成" << endl;
    }
    
    // 可视化频域
    void visualizeFrequencyDomain(const string& filename) {
        Mat magnitudeSpectrum(height, width, CV_32F);
        
        // 计算幅度谱并进行对数变换
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double magnitude = abs(frequencyDomain[i][j]);
                magnitudeSpectrum.at<float>(i, j) = log(1 + magnitude);
            }
        }
        
        // 将低频部分移到中心（fftshift）
        Mat shifted = magnitudeSpectrum.clone();
        int cx = width / 2;
        int cy = height / 2;
        
        Mat q0(shifted, Rect(0, 0, cx, cy));
        Mat q1(shifted, Rect(cx, 0, cx, cy));
        Mat q2(shifted, Rect(0, cy, cx, cy));
        Mat q3(shifted, Rect(cx, cy, cx, cy));
        
        Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
        
        // 归一化到0-255范围
        normalize(shifted, shifted, 0, 255, NORM_MINMAX);
        shifted.convertTo(shifted, CV_8U);
        
        imwrite(filename, shifted);
        cout << "频域可视化已保存到: " << filename << endl;
    }
    
    // 执行IFFT重建图像
    Mat performIFFT() {
        cout << "正在执行IFFT..." << endl;
        
        // 复制频域数据
        vector<vector<complex<double>>> ifftData = frequencyDomain;
        fft2D(ifftData, true);
        
        // 转换回图像
        Mat reconstructed(height, width, CV_8U);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double real_part = real(ifftData[i][j]);
                reconstructed.at<uchar>(i, j) = saturate_cast<uchar>(real_part);
            }
        }
        
        // 裁剪到原始大小
        Rect roi(0, 0, originalImage.cols, originalImage.rows);
        Mat result = reconstructed(roi);
        
        cout << "IFFT完成" << endl;
        return result;
    }
    
    // 低通滤波
    Mat applyLowPassFilterCentered(double cutoffRatio = 0.1) {
    cout << "正在应用中心化低通滤波..." << endl;
    
    // 复制频域数据
    vector<vector<complex<double>>> filteredData = frequencyDomain;
    
    // 计算截止半径
    double cutoffRadius = min(width, height) * cutoffRatio / 2.0;
    
    // 应用低通滤波器 - 从中心计算距离
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // 计算到中心的距离（考虑周期性）
            int di = min(i, height - i);
            int dj = min(j, width - j);
            double distance = sqrt(di * di + dj * dj);
            
            if (distance > cutoffRadius) {
                filteredData[i][j] = complex<double>(0, 0);
            }
        }
    }
    
    // 执行IFFT
    fft2D(filteredData, true);
    
    // 转换回图像
    Mat filtered(height, width, CV_8U);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double real_part = real(filteredData[i][j]);
            // 确保值在有效范围内
            real_part = max(0.0, min(255.0, real_part));
            filtered.at<uchar>(i, j) = saturate_cast<uchar>(real_part);
        }
    }
    
    // 裁剪到原始大小
    Rect roi(0, 0, originalImage.cols, originalImage.rows);
    Mat result = filtered(roi);
    
    cout << "中心化低通滤波完成，截止半径: " << cutoffRadius << endl;
    return result;
}
    
    // 验证FFT/IFFT可逆性
    void verifyReversibility() {
        cout << "\n=== 验证FFT/IFFT可逆性 ===" << endl;
        
        Mat reconstructed = performIFFT();
        
        double mse = calculateMSE(originalImage, reconstructed);
        double psnr = calculatePSNR(originalImage, reconstructed);
        
        cout << "重建图像 MSE: " << mse << endl;
        cout << "重建图像 PSNR: " << psnr << " dB" << endl;
        
        // 保存到output文件夹
        string filename = outputDir + "reconstructed_image.png";
        imwrite(filename, reconstructed);
        cout << "重建图像已保存到: " << filename << endl;
        
        if (mse < 1e-10) {
            cout << "✓ FFT/IFFT具有完美的可逆性" << endl;
        } else if (psnr > 50) {
            cout << "✓ FFT/IFFT具有良好的可逆性" << endl;
        } else {
            cout << "⚠ FFT/IFFT可逆性存在一定误差" << endl;
        }
    }
    
    // 测试低通滤波效果
    void testLowPassFilter() {
    cout << "\n=== 测试低通滤波效果 ===" << endl;
    
    vector<double> cutoffRatios = {0.05, 0.1, 0.2, 0.3, 0.5};

    for (double ratio : cutoffRatios) {
        // 使用中心化方法
        Mat filtered = applyLowPassFilterCentered(ratio);
        
        double mse = calculateMSE(originalImage, filtered);
        double psnr = calculatePSNR(originalImage, filtered);
        
        cout << "\n截止比例 " << ratio << ":" << endl;
        cout << "  MSE: " << mse << endl;
        cout << "  PSNR: " << psnr << " dB" << endl;
        
        // 检查图像是否为空
        Scalar meanVal = mean(filtered);
        cout << "  平均像素值: " << meanVal[0] << endl;
        
        // 保存到output文件夹
        string filename = outputDir + "filtered_" + to_string(int(ratio * 100)) + ".png";
        imwrite(filename, filtered);
        cout << "  滤波图像已保存到: " << filename << endl;
    }
}
    
    // 生成详细报告
    void generateReport() {
        // 保存到output文件夹
        string reportFilename = outputDir + "fft_analysis_report.txt";
        ofstream report(reportFilename);
        
        report << "FFT图像处理分析报告\n";
        report << "==================\n\n";
        
        report << "原始图像信息:\n";
        report << "- 尺寸: " << originalImage.cols << "x" << originalImage.rows << "\n";
        report << "- 处理尺寸: " << width << "x" << height << "\n\n";
        
        // 重建图像分析
        Mat reconstructed = performIFFT();
        double mse_recon = calculateMSE(originalImage, reconstructed);
        double psnr_recon = calculatePSNR(originalImage, reconstructed);
        
        report << "FFT/IFFT可逆性分析:\n";
        report << "- MSE: " << mse_recon << "\n";
        report << "- PSNR: " << psnr_recon << " dB\n";
        report << "- 结论: " << (psnr_recon > 50 ? "优秀" : "一般") << "\n\n";
        
        // 低通滤波分析
        report << "低通滤波效果分析:\n";
        vector<double> ratios = {0.05, 0.1, 0.2, 0.3};
        for (double ratio : ratios) {
            Mat filtered = applyLowPassFilterCentered(ratio);
            double mse = calculateMSE(originalImage, filtered);
            double psnr = calculatePSNR(originalImage, filtered);
            
            report << "- 截止比例 " << ratio << ": MSE=" << mse 
                   << ", PSNR=" << psnr << " dB\n";
        }
        
        report.close();
        cout << "\n分析报告已保存到: " << reportFilename << endl;
    }
};

int main() {
    FFTImageProcessor processor;
    
    // 读取图像
    if (!processor.loadImage("data/input_image.png")) {
        cout << "请确保图像文件存在于data文件夹中并重新运行程序" << endl;
        return -1;
    }

    // if (!processor.loadImage("data/miku.png")) {
    //     cout << "请确保图像文件存在于data文件夹中并重新运行程序" << endl;
    //     return -1;
    // }
    
    // 执行FFT
    processor.performFFT();
    
    // 可视化频域（保存到output文件夹）
    processor.visualizeFrequencyDomain("output/frequency_spectrum.png");
    
    // 验证FFT/IFFT可逆性
    processor.verifyReversibility();
    
    // 测试低通滤波
    processor.testLowPassFilter();
    
    // 生成报告
    processor.generateReport();
    
    cout << "\n所有任务完成！" << endl;
    cout << "生成的文件:" << endl;
    cout << "- output/frequency_spectrum.png (频域可视化)" << endl;
    cout << "- output/reconstructed_image.png (重建图像)" << endl;
    cout << "- output/filtered_*.png (不同程度的低通滤波结果)" << endl;
    cout << "- output/fft_analysis_report.txt (详细分析报告)" << endl;
    
    return 0;
}