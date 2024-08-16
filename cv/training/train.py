#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Training for YOLO to find distance from Glove to object to grasp
#
###

from ultralytics import YOLO
from torchinfo import summary
import logging
import sys
import os

from torch import cuda, backends

## Configuration # From MICLab
sys.path.insert(0, '..')
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), '../config.yaml'))
configs = config.get_config()

device = "cpu" 
if cuda.is_available(): device = "cuda" 
if backends.mps.is_available() and backends.mps.is_built(): device = "mps"

## Logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

logger.info(f"setup: Device = {device}")

# Model settings
image_sz = max( configs['training']['imageSize'])
dataSet = configs['training']['dataSet']
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/
modelFile = configs['training']['modelsDir'] +'/' + configs['training']['modelFile']
weightsFile = configs['training']['weightsDir'] +'/' + configs['training']['weightsFile']

#HyperPerams
epochs = configs['training']['epochs']


logger.info(f"Image Size: {image_sz}")

# From: https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/#freeze-backbone
transLearn = configs['training']['transLearn']
if transLearn:
       yoloModel = YOLO(modelFile).load(weightsFile)  # build from YAML and transfer weights
       freezeLayer = configs['training']['freezeLayer'] # First 10 layers are the backbone (10: freezes 0-9)

       freeze = [f"model.{x}." for x in range(freezeLayer)]  # which layers to freeze
       logger.info(f"Layers to freeze: {freeze}")
       logger.info("--------------------------------------")
       for k, v in yoloModel.named_parameters():
              #logger.info(f"k: {k}")
              v.requires_grad = True
              if any(x in k for x in freeze):
                     logger.info(f"Freezing layer {k}")
                     v.requires_grad = False
elif configs['training']['transLearn']:
       yoloModel = YOLO(modelFile)
else:
       yoloModel = YOLO(weightsFile)

yoloModel.info(detailed=True)
'''
image_depth = 3
modelSum = summary(model=yoloModel.model, 
                   #verbose=2,
                   input_size=(1, image_depth, image_sz, image_sz), # make sure this is "input_size", not "input_shape": must be square
            #col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "trainable"], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
            #col_width=20,
            row_settings=["var_names"]
            )
'''

results = yoloModel.train(hsv_h=1.0, plots=True, pretrained=transLearn, data=dataSet, epochs=epochs, imgsz=image_sz, device=device) # cpu, cuda, mps