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
# must run python 3.11
#
# Must run under Linux
# Note: takes a bloody long time!
#
# Files created:
#
###
from pathlib import Path
from ultralytics import YOLO

# From MICLab
import sys, os
sys.path.insert(0, '..')  # ConfigParser is one dir lower
from ConfigParser import ConfigParser
## Configuration
config = ConfigParser(os.path.join(os.getcwd(), '../config.yaml'))
configs = config.get_config()

def saveModel(modelFile, dataSet, imgH, imgW ):

    # Load the state dict
    #model.load_state_dict(torch.load(modelFile), strict=False) 
    model = YOLO(modelFile)  # build a new model from YAML

    '''
    must use python 3.11 for the exporter to work as of 7/7/24
    '''
    #model.export(format="tflite", data=dataSet, imgsz=(imgH, imgW), int8=True)
    # https://github.com/ultralytics/ultralytics/issues/1185  #I did not seem to have a problem tho
    model.export(format="edgetpu", data=dataSet, imgsz=(imgH, imgW), int8=True) #Edge TPU is linux only
    #Img is H, w



imgH = configs['training']['imageSize'][0]
imgW = configs['training']['imageSize'][1]
weightsDir = configs['training']['weightsDir']
weightsFile = configs['training']['weightsFile']
dataSet = configs['training']['dataSet']

modelPath = Path(weightsDir)
modelFile = modelPath/weightsFile

saveModel(modelFile=modelFile, dataSet=dataSet, imgH=imgH, imgW=imgW) 
