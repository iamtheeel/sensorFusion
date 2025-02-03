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

from threading import Thread

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

#Define the cameras
from camera import camera
inputCam_1 = camera(configs, configs['runTime']['camId'])
if(configs['runTime']['nCameras'] == 2):
    inputCam_2 = camera(configs, configs['runTime']['camId_2'])


if device == "tpu":
    runTimeCheckThread = True
    gpioPin = configs['timeSync']['gpio_pin']
    from periphery import GPIO  #pip install python-periphery
    timeTrigerGPIO = GPIO(gpioPin, "in")
    timeTrigerGPIO.edge = "raising" #“none”, “rising”, “falling”, or “both”

    logger.info(f"GPIO pin {gpioPin} interupt support = {timeTrigerGPIO.supports_interrupts}")

def checkClockReset_thread():
    logger.info(f"Starting clock reset thread: GPIO {timeTrigerGPIO}")
    while runTimeCheckThread:
        timeTrigerGPIO.poll() #Wait for the edige
        inputCam_1.setZeroTime()
        if(configs['runTime']['nCameras'] == 2):
            inputCam_2.setZeroTime()


## Import our items after we set the log leve
from modelRunTime import modelRunTime
import distance
import display


runCam = [True]* configs['runTime']['nCameras'] 
def get_cam1Image(cam):
    logger.info(f"Init Camera Thread")
    while runCam[0]:
        cam.capImage()
def get_cam2Image(cam):
    logger.info(f"Init Camera 2 Thread")
    while runCam[1]:
        cam.capImage()




def sanitizeStr(str):
    import re
    str = re.sub(r"[^\w\s]", "-", str) # Remove special chars
    str = re.sub(r"\s+", "-", str) #remove white space
    return str


def handleImage(image, imgCapTime, dCalc, objDisp, camId = 1 ):
    logger.info(f"------------------Camera {camId}---------------------------")
    #logger.info(f"size: {image_1.shape}")

    if(configs['debugs']['runInfer']):
        results = infer.runInference(image)
        validRes = dCalc.loadData(results )

        # Send the results over serial
        # make object from serialComms.py
        logger.info(f"Image Capture Time: {imgCapTime}")
        # $24, "CV", imgCapTime (uint_32), handConf (uint8), object class (uint8), object conf (uint8), Distance (uint16), <LF><CR>
    else: 
        validRes = False

    if configs['debugs']['dispResults']:
        # Show the image
        exitStatus = objDisp.draw(image, dCalc, validRes, camId, imageFile)
        #exitStatus = True

        if exitStatus == ord('q'):  # q = 113
            return False
            logger.info(f"********   quit now ***********")
        
    return True

if __name__ == "__main__":
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
        ## Load the camera
        camThread_1 = Thread(target=get_cam1Image, args=(inputCam_1, ))
        camThread_1.start()
        if(configs['runTime']['nCameras'] == 2):
            camThread_2 = Thread(target=get_cam2Image, args=(inputCam_2, ))
            camThread_2.start()

        startTime = [time.time()] * configs['runTime']['nCameras'] 
        dataRateTime = [0] * configs['runTime']['nCameras']  
        frameTime = 1/configs['runTime']['camRateHz']

        if device == "tpu":
            getTimeSetThread = Thread(target=checkClockReset_thread)
            getTimeSetThread.start()


        # Get the image
        while all(runCam):
            thisTime = time.time()
            camStat = [False, False]
            #camStat = [False]*configs['runTime']['nCameras'] 
            dataRateTime[0] = (thisTime-startTime[0])
            if(dataRateTime[0]) >= frameTime: 
                #logger.info(f"Get next image")
                camStat[0], image_1, camTime_1 = inputCam_1.getImage()

            if(configs['runTime']['nCameras'] == 2):
                dataRateTime[1] = (thisTime-startTime[1])
                if(dataRateTime[1]) >= frameTime: 
                    camStat[1], image_2, camTime_2 = inputCam_2.getImage()

            if camStat[0]:
                runCam[0] = handleImage(image_1, camTime_1, distCalc, handObjDisp, camId=1)
                logger.info(f"Total cam 1 time: {dataRateTime[0]*1000:.2f}ms, {1/dataRateTime[0]:.1f}Hz")
                startTime[0] = time.time()

            if camStat[1]:
                runCam[1] = handleImage(image_2, camTime_2, distCalc_2, handObjDisp_2, camId=2)
                logger.info(f"Total cam 2 time: {dataRateTime[1]*1000:.2f}ms, {1/dataRateTime[1]:.1f}Hz")
                startTime[1] = time.time()


            if(configs['runTime']['displaySettings']['runCamOnce']): 
                logger.info(f"Exit after one shot")

            
        # Destructor
        runCam = [False, False] # Kill both cameras
        camThread_1.join() # join the thread back to main
        del inputCam_1 
        if(configs['runTime']['nCameras'] == 2):
            camThread_2.join() # join the thread back to main
            del inputCam_2 

        
        if device == "tpu":
            runTimeCheckThread = False
            getTimeSetThread.join()
            timeTrigerGPIO.close()


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

                validRes = distCalc.loadData(results )
                if configs['debugs']['dispResults']:
                    exitStatus = handObjDisp.draw(thisImgFile, distCalc, validRes)
                    if exitStatus == ord('q'):  # q = 113
                        logger.info(f"********   quit now ***********")
                        exit()

    else: # single image
        image = configs['runTime']['imageDir'] + '/' + configs['runTime']['imgSrc']
        results = infer.runInference(image)

        validRes = distCalc.loadData(results)
        if configs['debugs']['dispResults']:
            handObjDisp.draw(image, distCalc, validRes)
    
