# sensorFusion on Raspberry Pi 5

This guide explains how to run the sensorFusion computer vision system on Raspberry Pi 5, replacing the TPU-specific code with CPU/GPU alternatives.

## Overview

The Raspberry Pi 5 version of sensorFusion uses:
- **TensorFlow Lite Runtime** for optimized inference on ARM architecture
- **ONNX Runtime** as an alternative inference engine
- **OpenVINO** for optional GPU acceleration
- **Multi-threading** for better CPU performance

## Hardware Requirements

- **Raspberry Pi 5** (4GB or 8GB RAM recommended)
- **Camera module** (Pi Camera or USB webcam)
- **MicroSD card** (32GB+ recommended, Class 10 or better)
- **Power supply** (5V/3A minimum)
- **Optional**: USB webcam for additional camera input

## Software Requirements

- **Raspberry Pi OS** (64-bit recommended)
- **Python 3.9+**
- **TensorFlow Lite Runtime**
- **ONNX Runtime**
- **OpenCV**

## Quick Setup

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/iamtheeel/sensorFusion.git
cd sensorFusion/cv/running

# Make setup script executable
chmod +x setup_rpi.sh

# Run the setup script
bash setup_rpi.sh
```

### 2. Reboot

```bash
sudo reboot
```

### 3. Test Installation

```bash
cd sensorFusion/cv/running
python3 test_rpi.py
```

### 4. Run the Application

```bash
python3 runImage.py
```

## Manual Installation

If you prefer manual installation:

### 1. System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtcore4 libqtgui4 libqt4-test
sudo apt install -y libgstreamer1.0-0 libgstreamer-plugins-base1.0-0
sudo apt install -y libgtk-3-0 libavcodec-dev libavformat-dev
sudo apt install -y libswscale-dev libv4l-dev libxvidcore-dev
sudo apt install -y libx264-dev libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y gfortran libopenblas-dev liblapack-dev
```

### 2. Enable Interfaces

```bash
# Enable camera
sudo raspi-config nonint do_camera 0

# Enable I2C (for servos)
sudo raspi-config nonint do_i2c 0
```

### 3. Python Dependencies

```bash
pip3 install --upgrade pip setuptools wheel
pip3 install tflite-runtime onnxruntime
pip3 install -r requirements_rpi.txt
```

### 4. Performance Optimization

```bash
# Add to ~/.bashrc
echo "export OPENBLAS_NUM_THREADS=4" >> ~/.bashrc
echo "export OMP_NUM_THREADS=4" >> ~/.bashrc
echo "export MKL_NUM_THREADS=4" >> ~/.bashrc

# Enable hardware acceleration
sudo echo "gpu_mem=128" >> /boot/config.txt
sudo echo "dtoverlay=vc4-kms-v3d" >> /boot/config.txt
```

## Model Export for Raspberry Pi

Your trained models need to be exported to TFLite or ONNX format for optimal performance on Raspberry Pi.

### Export to TensorFlow Lite

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/trained/model.pt')

# Export to TFLite
model.export(format='tflite', int8=True, imgsz=640)
```

### Export to ONNX

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/trained/model.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640)
```

## Configuration

Edit `config.yaml` to configure Raspberry Pi specific settings:

```yaml
runTime:
    # Raspberry Pi specific settings
    rpi_use_gpu: False  # Set to True for GPU acceleration
    rpi_num_threads: 4  # Number of CPU threads
    
    # Camera settings
    camId: 0  # Camera device ID
    camRateHz: 5  # Frame rate
    
    # Model settings
    distSettings:
        handThreshold: 0.5
        objectThreshold: 0.5
        nmsIouThreshold: 0.45
        handClass: 80
```

## Performance Optimization

### 1. CPU Optimization

- **Multi-threading**: Set `rpi_num_threads` to match your CPU cores (4 for Pi 5)
- **Environment variables**: Set thread limits for numerical libraries
- **Model quantization**: Use int8 quantized models for faster inference

### 2. Memory Optimization

- **Reduce image size**: Use smaller input sizes (e.g., 320x320 instead of 640x640)
- **Batch processing**: Process one frame at a time
- **Garbage collection**: Monitor memory usage

### 3. GPU Acceleration (Experimental)

Enable GPU acceleration by setting `rpi_use_gpu: True` in config.yaml. This requires:
- OpenVINO installation
- Compatible model format
- Sufficient GPU memory

## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Check camera interface
   sudo raspi-config nonint do_camera 0
   # Reboot and test
   vcgencmd get_camera
   ```

2. **Slow inference**
   - Reduce image size in config.yaml
   - Use quantized models (int8)
   - Increase number of threads
   - Check CPU temperature and throttling

3. **Memory errors**
   - Reduce image size
   - Close unnecessary applications
   - Monitor memory usage: `free -h`

4. **Model loading errors**
   - Ensure model is in TFLite or ONNX format
   - Check model file path in config.yaml
   - Verify model compatibility

### Performance Monitoring

```bash
# Monitor CPU usage
htop

# Monitor memory usage
free -h

# Monitor temperature
vcgencmd measure_temp

# Monitor GPU memory
vcgencmd get_mem gpu
```

## Expected Performance

On Raspberry Pi 5 (4GB RAM):

| Model Size | Input Size | Inference Time | FPS |
|------------|------------|----------------|-----|
| YOLOv8n | 320x320 | ~150ms | ~6.7 |
| YOLOv8n | 640x640 | ~400ms | ~2.5 |
| YOLOv8s | 320x320 | ~300ms | ~3.3 |
| YOLOv8s | 640x640 | ~800ms | ~1.25 |

*Performance may vary based on system load and configuration*

## Advanced Configuration

### Custom Model Loading

You can specify different models for different use cases:

```yaml
training:
    weightsFile_rpi: "your_custom_model.tflite"  # Raspberry Pi specific model
    weightsFile_tpu: "your_tpu_model.tflite"     # TPU model (fallback)
```

### Multi-Camera Setup

Configure multiple cameras:

```yaml
runTime:
    nCameras: 2
    camId: 0      # Primary camera
    camId_2: 1    # Secondary camera
```

### Custom Inference Settings

```yaml
runTime:
    rpi_use_gpu: True
    rpi_num_threads: 4
    distSettings:
        handThreshold: 0.6
        objectThreshold: 0.7
        nmsIouThreshold: 0.5
```

## Support

For issues specific to Raspberry Pi 5:

1. Check the test script output: `python3 test_rpi.py`
2. Review system logs: `dmesg | tail -20`
3. Monitor resource usage during operation
4. Ensure proper cooling and power supply

## Migration from TPU

If migrating from Coral Dev Board:

1. **Export models** to TFLite/ONNX format
2. **Update config.yaml** with Raspberry Pi settings
3. **Test with test_rpi.py** before running main application
4. **Adjust performance settings** based on your requirements
5. **Update camera settings** if using different camera hardware

## License

This Raspberry Pi adaptation maintains the same license as the original sensorFusion project. 