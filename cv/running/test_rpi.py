#!/usr/bin/env python3
#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Raspberry Pi 5 Test Script
# Tests the installation and basic functionality
#
###

import sys
import os
import platform
import logging

# Add the project root to the path
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_system_info():
    """Test system information"""
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    
    # Check if we're on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                print("✓ Raspberry Pi detected")
                return True
            else:
                print("✗ Not running on Raspberry Pi")
                return False
    except:
        print("✗ Could not read CPU info")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("\n=== Testing Imports ===")
    
    packages = [
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
        ('tqdm', 'tqdm'),
        ('pyyaml', 'yaml'),
        ('Pillow', 'PIL'),
        ('matplotlib', 'matplotlib'),
    ]
    
    all_good = True
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            all_good = False
    
    # Test TensorFlow Lite
    try:
        import tflite_runtime.interpreter as tflite
        print("✓ TensorFlow Lite Runtime")
    except ImportError as e:
        print(f"✗ TensorFlow Lite Runtime: {e}")
        all_good = False
    
    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        print("✓ ONNX Runtime")
    except ImportError as e:
        print(f"✗ ONNX Runtime: {e}")
        all_good = False
    
    return all_good

def test_camera():
    """Test camera access"""
    print("\n=== Testing Camera ===")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working (frame size: {frame.shape})")
                cap.release()
                return True
            else:
                print("✗ Camera opened but couldn't read frame")
                cap.release()
                return False
        else:
            print("✗ Could not open camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_model_loading():
    """Test model loading"""
    print("\n=== Testing Model Loading ===")
    
    try:
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        # Check if model files exist
        weights_dir = configs['training']['weightsDir']
        weights_file = configs['training'].get('weightsFile_rpi', configs['training']['weightsFile_tpu'])
        model_path = os.path.join(weights_dir, weights_file)
        
        if os.path.exists(model_path):
            print(f"✓ Model file found: {model_path}")
            
            # Try to load the model
            try:
                from rpiModel import RaspberryPiModel
                dataSetFile = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
                
                model = RaspberryPiModel(
                    model_path, 
                    dataSetFile,
                    conf_thresh=0.25,
                    iou_thresh=0.45,
                    v8=True,
                    use_gpu=False,
                    num_threads=4
                )
                print("✓ Model loaded successfully")
                return True
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                return False
        else:
            print(f"✗ Model file not found: {model_path}")
            print("  You need to export your trained model to TFLite or ONNX format")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_inference():
    """Test basic inference"""
    print("\n=== Testing Inference ===")
    
    try:
        import numpy as np
        from rpiModel import RaspberryPiModel
        
        # Load configuration
        config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
        configs = config.get_config()
        
        weights_dir = configs['training']['weightsDir']
        weights_file = configs['training'].get('weightsFile_rpi', configs['training']['weightsFile_tpu'])
        model_path = os.path.join(weights_dir, weights_file)
        dataSetFile = configs['training']['dataSetDir'] + '/' + configs['training']['dataSet']
        
        if not os.path.exists(model_path):
            print("✗ Model file not found, skipping inference test")
            return False
        
        # Load model
        model = RaspberryPiModel(
            model_path, 
            dataSetFile,
            conf_thresh=0.25,
            iou_thresh=0.45,
            v8=True,
            use_gpu=False,
            num_threads=4
        )
        
        # Create dummy input
        input_size = model.get_image_size()
        dummy_input = np.random.random((3, *input_size)).astype(np.float32)
        
        # Run inference
        import time
        start_time = time.time()
        result = model.forward(dummy_input)
        inference_time = time.time() - start_time
        
        print(f"✓ Inference successful")
        print(f"  Input size: {input_size}")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  FPS: {1/inference_time:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Raspberry Pi 5 Test Suite for sensorFusion")
    print("=" * 50)
    
    tests = [
        ("System Information", test_system_info),
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("Model Loading", test_model_loading),
        ("Inference", test_inference),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your Raspberry Pi 5 setup is ready.")
        print("\nYou can now run the main application:")
        print("  python3 runImage.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Run the setup script: bash setup_rpi.sh")
        print("2. Reboot your Raspberry Pi: sudo reboot")
        print("3. Export your model to TFLite/ONNX format")
        print("4. Check camera permissions and connections")

if __name__ == "__main__":
    main() 