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
# 3.12 does not work, 3.9 does not work. Must be 3.11
#
# Must run under Linux
# But not on the server... not that Linux... er, no clue why
# Server Error:
#       AssertionError arument 'int8' is not supported for format='edgecpu'
#   Does not work with: ultralytics 8.3.92
#   Works on : 8.2.51
#   Might work if we remove the int8, as the edgeTpu is full_integer
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


imgSZ = max(configs['training']['imageSize']) #must be square
weightsDir = configs['training']['weightsDir']
weightsFile = configs['training']['weightsFile']
dataSet = f"{configs['training']['dataSetDir']}/{configs['training']['dataSet']}"

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
###Ramhog, running on 640px takes ~20gb of ram, and the ram error is not clear##
#model.export(format="edgetpu", data=dataSet, imgsz=imgSZ, int8=True) #Edge TPU is linux only #Img is H, w
## Very particular about version, works with ulralytics v8.2.51, not with 8.3.92
model.export(format="edgetpu", data=dataSet, imgsz=imgSZ) #Edge TPU is linux only #Img is H, w
# Up to 30GB now! Cut the number of classes down and its 'only' ~15GB
tend = time.time()
logger.info(f"Export time: {tstart-tend}")
