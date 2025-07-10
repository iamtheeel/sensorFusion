#!/bin/bash

# Raspberry Pi 5 Setup Script for sensorFusion
# This script installs all necessary dependencies for running YOLO models on Raspberry Pi 5

echo "Setting up sensorFusion for Raspberry Pi 5..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtcore4 \
    libqtgui4 \
    libqt4-test \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Enable I2C interface (for servos)
echo "Enabling I2C interface..."
sudo raspi-config nonint do_i2c 0

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install TensorFlow Lite Runtime for ARM
echo "Installing TensorFlow Lite Runtime..."
pip3 install tflite-runtime

# Install ONNX Runtime for ARM
echo "Installing ONNX Runtime..."
pip3 install onnxruntime

# Install other Python packages
echo "Installing additional Python packages..."
pip3 install -r requirements_rpi.txt

# Optional: Install OpenVINO for GPU acceleration
echo "Installing OpenVINO (optional GPU acceleration)..."
pip3 install openvino

# Set up environment variables for better performance
echo "Setting up environment variables..."
echo "export OPENBLAS_NUM_THREADS=4" >> ~/.bashrc
echo "export OMP_NUM_THREADS=4" >> ~/.bashrc
echo "export MKL_NUM_THREADS=4" >> ~/.bashrc

# Enable hardware acceleration
echo "Enabling hardware acceleration..."
sudo echo "gpu_mem=128" >> /boot/config.txt
sudo echo "dtoverlay=vc4-kms-v3d" >> /boot/config.txt

# Create output directory
echo "Creating output directories..."
mkdir -p out
mkdir -p logs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Reboot your Raspberry Pi: sudo reboot"
echo "2. Test the installation: python3 test_rpi.py"
echo "3. Run the main application: python3 runImage.py"
echo ""
echo "Note: You may need to export your trained models to TFLite or ONNX format"
echo "for optimal performance on Raspberry Pi 5." 