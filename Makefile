# Makefile for FFT Image Processor

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# 如果系统中是opencv2，请使用下面这行
# OPENCV_FLAGS = `pkg-config --cflags --libs opencv`

TARGET = fft_processor
SOURCE = fft_image_processor.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET) *.png *.txt

run: $(TARGET)
	./$(TARGET)

install_opencv_ubuntu:
	sudo apt update
	sudo apt install libopencv-dev

install_opencv_mac:
	brew install opencv

.PHONY: all clean run install_opencv_ubuntu install_opencv_mac
