#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Save the model to TensorFlow MicroControler
#
###
from pathlib import Path
from ultralytics import YOLO
import torch


def saveModel(model, imgLayers, imgWidth, imgHeight ):
    name = "best"
    # train13, v8  : 6.5MB
    # trainv14, 5.6: 8.9MB
    modelDir = "runs/detect/train21/weights/" 
    modelPath = Path(modelDir)
    fileName = name+".pt"
    modelFile = modelPath/fileName

    # Load the state dict
    #model = YOLO("models/yolov8n.yaml" )  # build a new model from YAML
    # v8: Opened as 12.1MB
    # v5.6: 17.4
    model.load_state_dict(torch.load(modelFile), strict=False) 

    # Save as ONNX
    #https://docs.ultralytics.com/modes/export/#arguments
    #model.export(format="onnx", imgsz=imgHeight, int8=True, , optimize=True)
    '''
    onnxFileName = name+".onnx"
    onnxFile = modelPath/onnxFileName
    print(f"Saveing to Onnx: {onnxFile}")
    input_shape = (1, imgLayers, imgWidth, imgHeight)
    #exit()
    torch.onnx.export(model, torch.randn(input_shape), onnxFile, verbose=True,
                      input_names=["input"],
                      output_names=["output"])
    '''

    # Convert to Tensorflow

    # Convert to TF Lite
    # barking: ModuleNotFoundError: No module named 'imp'
    # Run with python < 3.12: https://docs.python.org/3.11/library/imp.html
    # But it barks AFTER it saves the tfLite
    model.export(format="tflite", imgsz=imgHeight, int8=True, data="coco8.yaml", optimize=True) # saved as 3.2MB

    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i yolov8n_int8.tflite > yoloV8n_int8.h

model = YOLO("models/yolov8n.yaml" )  # build a new model from YAML
image_depth = 3
mean = 0
std = 0
saveModel(model, image_depth, 96, 96)