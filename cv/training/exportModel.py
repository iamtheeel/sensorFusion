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
# Note: takes a bloody long time!   Mostly mem ops, cpu stays low, looks hung, but is not
#
#
###
from pathlib import Path
from ultralytics import YOLO
import time

# From MICLab
import sys, os
sys.path.insert(0, '../..')  # ConfigParser is in the project root
from ConfigParser import ConfigParser
## Configuration
config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
configs = config.get_config()

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True


imgSZ = max(configs['training']['imageSize'][0], configs['training']['imageSize'][1]) #must be square
weightsDir = configs['training']['weightsDir']
weightsFile = configs['training']['weightsFile']
dataSet = configs['training']['dataSet']

modelPath = Path(weightsDir)
modelFile = modelPath/weightsFile

model = YOLO(modelFile)  # build a new model from YAML
#saveModel(modelFile=modelFile, dataSet=dataSet, imgH=imgH, imgW=imgW) 

'''
must use python 3.11 for the exporter to work as of 7/7/24
'''
#model.export(format="tflite", data=dataSet, imgsz=(imgH, imgW), int8=True)
# https://github.com/ultralytics/ultralytics/issues/1185  #I did not seem to have a problem tho
tstart = time.time()
model.export(format="edgetpu", data=dataSet, imgsz=imgSZ, int8=True) #Edge TPU is linux only #Img is H, w
###Ramhog, running on 640px takes ~20gb of ram, and the ram error is not clear##
tend = time.time()
logger.info(f"Export time: {tstart-tend}")
