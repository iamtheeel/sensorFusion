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
import os, sys
#import cv2
import time


# From MICLab
## Configuration
sys.path.insert(0, '../..')
from ConfigParser import ConfigParser
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
from modelRunTime import modelRunTime
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

def sanitizeStr(str):
    import re
    #str = str.replace(" ", "_")
    str = re.sub(r"[^\w\s]", "-", str) # Remove special chars
    str = re.sub(r"\s+", "-", str) #remove white space
    return str


if __name__ == "__main__":
    ## set the model information
    if(configs['debugs']['runInfer']):
        infer = modelRunTime(configs, device)
    
    distCalc = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])
    handObjDisp = display.displayHandObject(configs)
    
    ## Get image
    if configs['runTime']['imgSrc'] == 'camera':

        ## Set up the file saves
        imageFile = ""
        if(configs['debugs']['saveImages']):
            subject = sanitizeStr(input("Enter the subject ID: \n> "))
            object  = sanitizeStr(input("Enter the object: \n> "))
            run     = sanitizeStr(input("Enter the run (The run will start on <enter>): \n> "))

            logger.info(f"subject: {subject}")
            logger.info(f"object: {object}")
            logger.info(f"run: {run}")
            imageFile = f"{subject}_{object}_{run}"

        ## Load the camera
        inputCam = camera(configs)
        runThread = True
        camThread = Thread(target=getImage, args=(inputCam, ))
        camThread.start()

        startTime = time.time()
        endTime = startTime
        frameTime = 1/configs['runTime']['camRateHz']

        # Get the image
        runCam = True
        while runCam:
            endTime = time.time()
            camStat = False
            if(endTime - startTime) >= frameTime: 
                #logger.info(f"Get next image")
                camStat, image = inputCam.getImage()

            if camStat:
                logger.info("---------------------------------------------")
                logger.info(f"Image size: {image.shape}")

                if(configs['debugs']['runInfer']):
                    results = infer.runInference(image)
                    validRes = distCalc.loadData(results, device)
                else: 
                    validRes = False

                if configs['debugs']['dispResults']:
                    # Show the image
                    exitStatus = handObjDisp.draw(image, distCalc, validRes, imageFile)

                    if exitStatus == ord('q'):  # q = 113
                        runCam = False
                        logger.info(f"********   quit now ***********")

                #endTime = time.time()
                runTime = (endTime-startTime)
                logger.info(f"Total Loop time: {runTime*1000:.2f}ms, {1/runTime:.1f}Hz")
                startTime = endTime
            #else:
            #    logger.info(f"Image not ready: {configs['runTime']['camId']}")

            if(configs['runTime']['displaySettings']['runCamOnce']): 
                logger.info(f"Exit after one shot")
                runCam = False

            
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
    
