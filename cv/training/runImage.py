#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Run YOLO on a called image
#
###

from ultralytics import YOLO
import torch
from pathlib import Path

device = "cpu" 
if torch.cuda.is_available(): device = "cuda" 
if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"

# Configs
image_depth = 3
image_sz = 320 # Works
#thisImg = "datasets/combinedData/images/train/Dataset_apple_4.jpg"
# -- or --
image_dir = "datasets/combinedData/images/foo/"
weightsDir = "runs/detect/train51/weights/" #glass: YoloV3, imgsz=320, d,w: 
#model = YOLO("models/yolov3-tiny.yaml" )  # build a new model from YAML

name = "best"
modelPath = Path(weightsDir)
fileName = name+".pt"
modelFile = modelPath/fileName
print(f"model: {modelFile}")

# Load the state dict
model = YOLO(modelFile)  # 
#model.load_state_dict(torch.load(modelFile), strict=False) 

# Run the model
import os, fnmatch
listing = os.scandir(image_dir)
for thisFile in listing:
    if fnmatch.fnmatch(thisFile, '*.jpg'):
        thisImg = image_dir + "/" + thisFile.name
        results = model.predict(thisImg, imgsz=image_sz)
        for result in results:
            #result.show()
            print(result.boxes)