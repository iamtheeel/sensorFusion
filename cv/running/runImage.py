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
import cv2
import sys
import os
import time

# From MICLab
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser

# Internal
from modelRunTime import modelRunTime

## Configuration
config = ConfigParser(os.path.join(os.getcwd(), '../../config.yaml'))
configs = config.get_config()

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


## Import our items after we set the log leve
import distance
import display

from camera import camera
# From edgetpu

from threading import Thread
runThread = False
def getImage(cam):
    logger.info(f"Init Camera Thread")
    while runThread:
        cam.capImage()


if __name__ == "__main__":

    ## set the model information

    infer = modelRunTime(configs, device)
    
    distCalc = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])
    handObjDisp = display.displayHandObject(configs['runTime']['displaySettings'])
    
    ## Get image
    if configs['runTime']['imgSrc'] == 'camera':
        ## Load the camera
        inputCam = camera(configs)
        runThread = True
        camThread = Thread(target=getImage, args=(inputCam, ))
        camThread.start()

        # Get the image
        runCam = True
        while runCam:
            logger.info("---------------------------------------------")
            camStat, image = inputCam.getImage()

            if camStat:
                if configs['runTime']['displaySettings']['runCamOnce']: runCam = False

                #logger.info(f"Image size: {image.shape}")

                results = infer.runInference(image)
                validRes = distCalc.loadData(results, device)

                if configs['debugs']['dispResults']:
                    # Show the image
                    exitStatus = handObjDisp.draw(image, distCalc, validRes)

                    if exitStatus == ord('q'):  # q = 113
                        runCam = False
                        logger.info(f"********   quit now ***********")
            #else:
            #    logger.info(f"Image not ready: {configs['runTime']['camId']}")
                #camera.release()
                #camera = cv2.VideoCapture(configs['runTime']['camId'], cv2.CAP_ANY)
            
        # Destructor
        runThread = False
        camThread.join() # join the thread back to main
        del camera 

    elif configs['runTime']['imgSrc'] == 'directory':
        import os, fnmatch
        image_dir = configs['runTime']['imageDir']
        listing = os.scandir(image_dir)

        for thisFile in listing:
            #if fnmatch.fnmatch(thisFile, '*936.jpg'):
            if fnmatch.fnmatch(thisFile, '*.jpg'):
                #generate a list of files
                thisImgFile = image_dir + "/" + thisFile.name
                logger.info("---------------------------------------------")
                logger.info(f"File: {thisImgFile}")
                results = infer.runInference(thisImgFile)

                validRes = distCalc.loadData(results, device)
                if configs['debugs']['dispResults']:
                    exitStatus = handObjDisp.draw(thisImgFile, distCalc, validRes)
                    if exitStatus == ord('q'):  # q = 113
                        logger.info(f"********   quit now ***********")
                        exit()

    else: # single image
        image = configs['runTime']['imageDir'] + '/' + configs['runTime']['imgSrc']
        results = infer.runInference(image)

        validRes = distCalc.loadData(results, device)
        if configs['debugs']['dispResults']:
            handObjDisp.draw(image, distCalc, validRes)
    
