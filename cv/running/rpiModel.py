#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Raspberry Pi 5 Model Runtime
# Replaces TPU-specific code with CPU/GPU alternatives
#
###

import time
import os
import sys
import logging
import numpy as np
import cv2
import yaml
import json
from pathlib import Path

# Try to import TensorFlow Lite first, fallback to ONNX Runtime
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    logger.info("Using TensorFlow Lite runtime")
except ImportError:
    TFLITE_AVAILABLE = False
    logger.info("TensorFlow Lite not available, trying ONNX Runtime")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("Using ONNX Runtime")
except ImportError:
    ONNX_AVAILABLE = False
    logger.info("ONNX Runtime not available")

from nms import non_max_suppression, non_max_suppresion_v8
from utils import plot_one_box, Colors, get_image_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RaspberryPiModel")

class RaspberryPiModel:
    def __init__(self, model_file, names_file, conf_thresh=0.25, iou_thresh=0.45, 
                 filter_classes=None, agnostic_nms=False, max_det=1000, v8=False, 
                 use_gpu=False, num_threads=4):
        """
        Creates an object for running a YOLO model on Raspberry Pi 5
        
        Inputs:
          - model_file: path to tflite or onnx model file
          - names_file: yaml names file (yolov5 format)
          - conf_thresh: detection threshold
          - iou_thresh: NMS threshold
          - filter_classes: only output certain classes
          - agnostic_nms: use class-agnostic NMS
          - max_det: max number of detections
          - v8: YOLOv8 model flag
          - use_gpu: use GPU acceleration if available
          - num_threads: number of threads for CPU inference
        """
        
        model_file = os.path.abspath(model_file)
        self.model_file = model_file
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.filter_classes = filter_classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.v8 = v8
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        
        logger.info("Confidence threshold: {}".format(conf_thresh))
        logger.info("IOU threshold: {}".format(iou_thresh))
        logger.info("Using GPU: {}".format(use_gpu))
        logger.info("Number of threads: {}".format(num_threads))
        
        self.inference_time = None
        self.nms_time = None
        self.interpreter = None
        self.colors = Colors()
        
        self.get_names(names_file)
        self.make_interpreter()
        
    def get_names(self, path):
        """
        Load a names file
        
        Inputs:
          - path: path to names file in yaml format
        """
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
            
        names = cfg['names']
        logger.info("Loaded {} classes".format(len(names)))
        self.names = names
    
    def make_interpreter(self):
        """
        Internal function that loads the model and creates the interpreter
        """
        if self.model_file.endswith('.tflite'):
            self._load_tflite_model()
        elif self.model_file.endswith('.onnx'):
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_file}")
    
    def _load_tflite_model(self):
        """Load TensorFlow Lite model"""
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite runtime not available")
        
        try:
            # Create interpreter with optional GPU delegate
            if self.use_gpu:
                # Try to use GPU delegate if available
                try:
                    from tflite_runtime.interpreter import load_delegate
                    gpu_delegate = load_delegate('libedgetpu.so.1')
                    self.interpreter = tflite.Interpreter(
                        model_path=self.model_file,
                        experimental_delegates=[gpu_delegate]
                    )
                    logger.info("Using GPU delegate")
                except:
                    logger.warning("GPU delegate not available, falling back to CPU")
                    self.interpreter = tflite.Interpreter(model_path=self.model_file)
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_file)
            
            # Set number of threads for CPU inference
            self.interpreter.set_num_threads(self.num_threads)
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get quantization parameters
            self.input_zero = self.input_details[0]['quantization'][1]
            self.input_scale = self.input_details[0]['quantization'][0]
            self.output_zero = self.output_details[0]['quantization'][1]
            self.output_scale = self.output_details[0]['quantization'][0]
            
            # Handle non-quantized models
            if self.input_scale < 1e-9: self.input_scale = 1.0
            if self.output_scale < 1e-9: self.output_scale = 1.0
            
            # Get input size
            self.input_size = self.input_details[0]['shape'][1:3]  # Height, Width
            
            logger.info("Successfully loaded TensorFlow Lite model: {}".format(self.model_file))
            logger.info("Input size: {}".format(self.input_size))
            
        except Exception as e:
            logger.error(f"Failed to load TensorFlow Lite model: {e}")
            raise
    
    def _load_onnx_model(self):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        try:
            # Configure ONNX Runtime session options
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.num_threads
            session_options.inter_op_num_threads = self.num_threads
            
            # Set execution providers
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                try:
                    providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                    logger.info("Using OpenVINO GPU acceleration")
                except:
                    logger.warning("OpenVINO not available, using CPU")
            
            # Create ONNX Runtime session
            self.interpreter = ort.InferenceSession(
                self.model_file, 
                sess_options=session_options,
                providers=providers
            )
            
            # Get input/output details
            self.input_details = [{'name': input.name, 'shape': input.shape} 
                                for input in self.interpreter.get_inputs()]
            self.output_details = [{'name': output.name, 'shape': output.shape} 
                                 for output in self.interpreter.get_outputs()]
            
            # Get input size (assuming first input is the image)
            self.input_size = self.input_details[0]['shape'][2:4]  # Height, Width
            
            # For ONNX models, we'll use float32 (no quantization)
            self.input_zero = 0.0
            self.input_scale = 1.0
            self.output_zero = 0.0
            self.output_scale = 1.0
            
            logger.info("Successfully loaded ONNX model: {}".format(self.model_file))
            logger.info("Input size: {}".format(self.input_size))
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def get_image_size(self):
        """Get the input image size required by the model"""
        return self.input_size
    
    def predict(self, image_path, save_img=True, save_txt=True):
        """Predict on a single image file"""
        logger.info("Attempting to load {}".format(image_path))
        
        full_image, net_image, pad = get_image_tensor(image_path, self.input_size[0])
        pred = self.forward(net_image)
        
        base, ext = os.path.splitext(image_path)
        imgName = os.path.split(base)
        output_path = "out/" + imgName[1] + "_detect" + ext
        
        det = self.process_predictions(pred[0], full_image, pad, output_path, 
                                     save_img=save_img, save_txt=save_txt)
        return det
    
    def forward(self, x: np.ndarray, with_nms=True) -> np.ndarray:
        """
        Predict function using TensorFlow Lite or ONNX Runtime
        
        Inputs:
            x: (C, H, W) image tensor
            with_nms: apply NMS on output
            
        Returns:
            prediction array (with or without NMS applied)
        """
        tstart = time.time()
        
        # Transpose if C, H, W
        if x.shape[0] == 3:
            x = x.transpose((1, 2, 0))
        
        x = x.astype('float32')
        
        # Scale input for quantized models
        x = (x / self.input_scale) + self.input_zero
        
        # Prepare input based on model type
        if self.model_file.endswith('.tflite'):
            if self.v8:
                x = x[np.newaxis].astype(np.int8)
            else:
                x = x[np.newaxis].astype(np.uint8)
        else:  # ONNX
            x = x[np.newaxis].astype(np.float32)
        
        # Run inference
        if self.model_file.endswith('.tflite'):
            self.interpreter.set_tensor(self.input_details[0]['index'], x)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_details[0]['index']).copy()
        else:  # ONNX
            input_name = self.input_details[0]['name']
            raw_output = self.interpreter.run(None, {input_name: x})[0]
        
        # Scale output for quantized models
        result = (raw_output.astype('float32') - self.output_zero) * self.output_scale
        
        if self.v8:
            result = np.transpose(result, [0, 2, 1])  # transpose for yolov8 models
        
        self.inference_time = time.time() - tstart
        logger.info(f"inference time {self.inference_time}")
        
        if with_nms:
            tstart = time.time()
            if self.v8:
                nms_result = non_max_suppresion_v8(result, self.conf_thresh, self.iou_thresh, 
                                                 self.filter_classes, self.agnostic_nms, 
                                                 max_det=self.max_det)
            else:
                nms_result = non_max_suppression(result, self.conf_thresh, self.iou_thresh, 
                                               self.filter_classes, self.agnostic_nms, 
                                               max_det=self.max_det)
            self.nms_time = time.time() - tstart
            return nms_result
        else:
            return result
    
    def get_last_inference_time(self, with_nms=True):
        """Returns a tuple containing most recent inference and NMS time"""
        res = [self.inference_time]
        if with_nms:
            res.append(self.nms_time)
        return res
    
    def get_scaled_coords(self, boxes, output_image, pad):
        """Scale bounding boxes from model input size to output image size"""
        img_h, img_w = output_image.shape[:2]
        input_h, input_w = self.input_size
        
        # Remove padding
        pad_h, pad_w = pad
        
        # Scale coordinates
        boxes[:, [0, 2]] *= input_w / (input_w - 2 * pad_w)  # x coordinates
        boxes[:, [1, 3]] *= input_h / (input_h - 2 * pad_h)  # y coordinates
        
        # Adjust for padding
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        
        # Clip to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
        
        return boxes
    
    def process_predictions(self, det, output_image, pad, output_path="detection.jpg", 
                          save_img=True, save_txt=True, hide_labels=False, hide_conf=False):
        """Process predictions and optionally output an image with annotations"""
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = self.get_scaled_coords(det[:, :4], output_image, pad)
            output = {}
            base, ext = os.path.splitext(output_path)
            
            s = ""
            
            # Print results
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
            
            if s != "":
                s = s.strip()
                s = s[:-1]
            
            logger.info("Detected: {}".format(s))
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                    output_image = plot_one_box(xyxy, output_image, label=label, color=self.colors(c, True))
                if save_txt:
                    output[base] = {}
                    output[base]['box'] = xyxy
                    output[base]['conf'] = conf
                    output[base]['cls'] = cls
                    output[base]['cls_name'] = self.names[c]
                    
            if save_txt:
                output_txt = base + ".txt"
                print(f"Output file: {output_txt}")
                with open(output_txt, 'w') as f:
                    json.dump(output, f, indent=1)
            if save_img:
                print(f"Output image: {output_path}")
                cv2.imwrite(output_path, output_image)
            
        return det
    
    def exit(self):
        """Clean up resources"""
        logger.info("Exiting Raspberry Pi Model")
        # No specific cleanup needed for TFLite/ONNX 