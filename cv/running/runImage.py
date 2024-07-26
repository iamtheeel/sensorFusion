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

import numpy as np

import distance
import display

device = "cpu" 
if torch.cuda.is_available(): device = "cuda" 
if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"

# Configs
#image_dir = "datasets/combinedData/images/foo/"
image_dir = "datasets/testImages/"
weightsDir = "weights/" #Trained on server
#weightsDir = "runs/detect/train48/weights/" #glass: YoloV3, imgsz=320, d,w: 

name = "best"
modelPath = Path(weightsDir)
fileName = name+".pt"
modelFile = modelPath/fileName
print(f"model: {modelFile}")

# Load the state dict
model = YOLO(modelFile)  # 
#model.load_state_dict(torch.load(modelFile), strict=False) 

imagePxlPer_mm = 1.0
inferImgSize = [256, 320] # what is the image shape handed to inference
distCalc = distance.distanceCalculator(inferImgSize, imagePxlPer_mm)

#Color in BGR
lColor = [125, 125, 125]
hColor = [0, 255, 0]
oColor = [0, 0, 255]
handObjDisp = display.displayHandObject(hColor, oColor, lColor)

# Run the model
import os, fnmatch
listing = os.scandir(image_dir)
for thisFile in listing:
    if fnmatch.fnmatch(thisFile, '*.jpg'):
        thisImgFile = image_dir + "/" + thisFile.name

        results = model.predict(thisImgFile)
        for result in results:
            #print(result.boxes)
            #result.show()

            print("---------------------------------------------")
            print(f"Data: {result.boxes.data}")

            validRes = distCalc.loadData(result.boxes.data, result.boxes.cls)
            print(f"N objects detected: hands = {distCalc.nHands}, non hands = {distCalc.nNonHand}")
            print(f"Valid: {validRes}")
            if validRes:
                handObjDisp.draw(thisImgFile, distCalc)

                '''
    
                '''