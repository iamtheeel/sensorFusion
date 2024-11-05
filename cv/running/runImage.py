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
run_cam1Thread = False
def get_cam1Image(cam):
    logger.info(f"Init Camera Thread")
    while run_cam1Thread:
        cam.capImage()
run_cam2Thread = False
def get_cam2Image(cam):
    logger.info(f"Init Camera 2 Thread")
    while run_cam2Thread:
        cam.capImage()

def sanitizeStr(str):
    import re
    #str = str.replace(" ", "_")
    str = re.sub(r"[^\w\s]", "-", str) # Remove special chars
    str = re.sub(r"\s+", "-", str) #remove white space
    return str

TODO:
quit from multi cameras
save from multi cameras
timeing from multi cameras 

def handleImage(image, dCalc, objDisp, camId = 1 ):
    logger.info(f"------------------Camera {camId}---------------------------")
    #logger.info(f"size: {image_1.shape}")

    if(configs['debugs']['runInfer']):
        results = infer.runInference(image)
        validRes = dCalc.loadData(results, device)
    else: 
        validRes = False

    if configs['debugs']['dispResults']:
        # Show the image
        exitStatus = objDisp.draw(image, dCalc, validRes, imageFile)

        if exitStatus == ord('q'):  # q = 113
            return False
            logger.info(f"********   quit now ***********")
        
    return True

if __name__ == "__main__":
    ## set the model information
    if(configs['debugs']['runInfer']):
        infer = modelRunTime(configs, device)
    
    distCalc = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])
    handObjDisp = display.displayHandObject(configs)
    if(configs['runTime']['nCameras'] == 2):
        distCalc_2 = distance.distanceCalculator(configs['training']['imageSize'], 
                                           configs['runTime']['distSettings'])
        handObjDisp_2 = display.displayHandObject(configs, camNum=2)
    
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
        inputCam_1 = camera(configs, configs['runTime']['camId'])
        run_cam1Thread = True
        camThread_1 = Thread(target=get_cam1Image, args=(inputCam_1, ))
        camThread_1.start()
        if(configs['runTime']['nCameras'] == 2):
            inputCam_2 = camera(configs, configs['runTime']['camId_2'])
            run_cam2Thread = True
            camThread_2 = Thread(target=get_cam2Image, args=(inputCam_2, ))
            camThread_2.start()


        startTime = time.time()
        endTime = startTime
        frameTime = 1/configs['runTime']['camRateHz']

        # Get the image
        runCam = True # a q from either window exits
        while runCam:
            endTime = time.time()
            camStat_1 = False
            camStat_2 = False
            if(endTime - startTime) >= frameTime: 
                #logger.info(f"Get next image")
                camStat_1, image_1 = inputCam_1.getImage()

                if(configs['runTime']['nCameras'] == 2):
                    camStat_2, image_2 = inputCam_2.getImage()
                    print(f"Get cam 2: {camStat_2}")

            if camStat_1:
                runCam = handleImage(image_1, distCalc, handObjDisp, camId=1)
            #else:
            #    logger.info(f"Image not ready: {configs['runTime']['camId']}")
                ## TODO: Fixe the timing to take account two cameras
                runTime = (endTime-startTime)
                logger.info(f"Total Loop time: {runTime*1000:.2f}ms, {1/runTime:.1f}Hz")
             
                startTime = endTime
            if camStat_2:
                print(f" ********    cam 2 ********")
                runCam = handleImage(image_2, distCalc_2, handObjDisp_2, camId=2)


            if(configs['runTime']['displaySettings']['runCamOnce']): 
                logger.info(f"Exit after one shot")
                runCam = False

            
        # Destructor
        run_cam1Thread = False
        camThread_1.join() # join the thread back to main
        del inputCam_1 
        if(configs['runTime']['nCameras'] == 2):
            run_cam2Thread = False
            camThread_2.join() # join the thread back to main
            del inputCam_2 


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
    
