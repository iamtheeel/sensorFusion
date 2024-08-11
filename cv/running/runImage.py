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

import platform
import logging
from pathlib import Path
import numpy as np
import cv2

# From MICLab
import sys
import os
sys.path.insert(0, '..')
from ConfigParser import ConfigParser

## Configuration
config = ConfigParser(os.path.join(os.getcwd(), '../config.yaml'))
configs = config.get_config()
#print(configs)
#print(configs['logLevel'])

## Logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

## What platform are we running on
machine = platform.machine()
logger.info(f"machine: {machine}")

if machine == "aarch64":
    device = "tpu"
else:
    import torch
    device = "cpu" 
    if torch.cuda.is_available(): device = "cuda" 
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"

if device != "tpu":
    from ultralytics import YOLO
else:
    from edgetpumodel import EdgeTPUModel

## Import our items after we set the log leve
import distance
import display

# From edgetpu
from utils import get_image_tensor


# Configs
## set the model information
image_dir = configs['runTime']['imageDir']

if device == "tpu":
    weightsFile = configs['training']['weightsFile_tpu']
else:
    weightsFile = configs['training']['weightsFile']
modelPath = Path(configs['training']['weightsDir'])
modelFile = modelPath/weightsFile
logger.info(f"model: {modelFile}")

# Load the state dict
if device != "tpu":
    model = YOLO(modelFile)  # 
else:
    model = EdgeTPUModel(modelFile, configs['training']['dataSet'], conf_thresh=0.1, iou_thresh=0.1, v8=True)
    input_size = model.get_image_size()
    x = (255*np.random.random((3,*input_size))).astype(np.int8)
    model.forward(x) # Prime with the image size


distCalc = distance.distanceCalculator(configs['training']['imageSize'], configs['runTime']['distSettings'])
handObjDisp = display.displayHandObject(configs['runTime']['displaySettings']) #hColor, oColor, lColor)

## Get image
# https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
camera = cv2.VideoCapture(configs['runTime']['camId'])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, configs['training']['imageSize'][0])
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  configs['training']['imageSize'][1])

# Get the image
camStat, image = camera.read()

# Precess the image
if device == "tpu":
    full_image, net_image, pad = get_image_tensor(image, input_size[0])
    pred = model.forward(net_image)
    model.process_predictions(pred[0], full_image, pad)
                
    tinference, tnms = model.get_last_inference_time()
    logger.info("Frame done in {}".format(tinference+tnms))
else: 
    results = model.predict(image) # Returns a dict


# Display the image
if configs['debugs']['dispResults']:
    logger.info(f"Cam Status: {camStat}")
    cv2.imshow("cameraTest", image)
    waitkey = cv2.waitKey()
exit()

# Run the model
import os, fnmatch
listing = os.scandir(image_dir)
for thisFile in listing:
    #if fnmatch.fnmatch(thisFile, '*936.jpg'):
    if fnmatch.fnmatch(thisFile, '*.jpg'):
        #generate a list of files
        thisImgFile = image_dir + "/" + thisFile.name

        if device == "tpu":
            # Returns a numpy array: x1, x2, y1, y2, conf, class
            results = model.predict(thisImgFile, save_img=configs['debugs']['showInfResults'] , save_txt=configs['debugs']['showInfResults'])
            inferTime = model.get_last_inference_time() #inference time, nms time
            logger.info(f"Inference Time: {inferTime}")
            #if results.shape[0] != 1:
            #    logger.info(results)
            logger.info(f"Results: {type(results)}, {results}")


        else:
            results = model.predict(thisImgFile) # Returns a dict
            if debug:
                #logger.info(f"Results: {type(results)}, {results}")
                #logger.info(f"Results[0]: {type(results[0])}, {results[0]}")
                #logger.info(f"Results shape: {type(results)}, {results.shape}")
                logger.info(f"Results.boxes: {type(results[0].boxes.data)}, {results[0].boxes.data}")
                logger.info(f"File: {thisImgFile}")

        logger.info("---------------------------------------------")
        if device == "tpu":
            boxes = results
                #if results.shape[0] != 1: # If we have detected more than one object
                #if result[5] != 80:  # not a hand (47 is apple)
                    #logger.info(f"result:{result}")
                    #logger.info("result:{}".format(result))
                    #validRes = distCalc.loadData(result)

        else:
            if configs['debugs']['showInfResults']:
                logger.info(results[0].boxes)
                results[0].show()

            boxes = results[0].boxes.data

            if debug:
                if device != "tpu":
                    #logger.info(f"Results: {type(results[0].boxes.data)}, {results[0].boxes.data}")
                    logger.info(f"Results: {type(results[0].boxes)}, {results[0].boxes}")

        validRes = distCalc.loadData(boxes, device)

        if debug:
            logger.info(f"N objects detected: hands = {distCalc.nHands}, non hands = {distCalc.nNonHand}")
            logger.info(f"Valid: {validRes}, distance: {distCalc.bestDist}")

        if configs['debugs']['dispResults']:
            exitStatus = handObjDisp.draw(thisImgFile, distCalc, validRes)
            logger.info(f"exitSatus: {exitStatus}: ")

            if exitStatus == ord('q'):  # q = 113
                logger.info(f"********   quit now ***********")
                exit()

