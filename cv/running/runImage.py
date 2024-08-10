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
machine = platform.machine()
print(f"machine: {machine}")

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
    import logging
    from edgetpumodel import EdgeTPUModel


from pathlib import Path

import numpy as np

import distance
import display

debug = True
showInfResults = True


# Configs
#image_dir = "../datasets/combinedData/images/val"
image_dir = "../datasets/testImages"

weightsDir = "../weights/" #Trained on server
#weightsDir = "runs/detect/train48/weights/" #glass: YoloV3, imgsz=320, d,w: 

#weightsFile = "scratch_320.pt" #Trained from scratch imgsz=320
#weightsFile = "yolov5nu.pt" #PreTrained yolo v5 nano 640px image size, 
if device != "tpu":
    weightsFile = "yolov5nu_transferFromCOCO.pt" #yolo v5 nano 640px image size, transfern learning from COCO\
else:
    weightsFile = "yolov5nu_transferFromCOCO_full_integer_quant_edgetpu_608.tflite" #yolo v5 nano 640px image size, transfern learning from COCO

# Display settings
imagePxlPer_mm = 1.0
handThreshold = 0.6
objectThreshold = 0.6
inferImgSize = [256, 320] # what is the image shape handed to inference
handClass = 80

modelPath = Path(weightsDir)
modelFile = modelPath/weightsFile
print(f"model: {modelFile}")

dataSet = "../datasets/coco_withHand.yaml"

# Load the state dict
if device != "tpu":
    model = YOLO(modelFile)  # 
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if debug == False:
        logging.disable(level=logging.CRITICAL)
        logger.disabled = True

    model = EdgeTPUModel(modelFile, dataSet, conf_thresh=0.1, iou_thresh=0.1, v8=True)
    input_size = model.get_image_size()
    x = (255*np.random.random((3,*input_size))).astype(np.int8)
    model.forward(x) # Prime with the image size


#Init the calculator
distCalc = distance.distanceCalculator(inferImgSize, imagePxlPer_mm, handThresh=handThreshold, objThresh=objectThreshold, handClass=handClass)

#Color in BGR
lColor = [125, 125, 125]
hColor = [0, 255, 0]
oColor = [0, 0, 255]
handObjDisp = display.displayHandObject(hColor, oColor, lColor)

# Run the model
import os, fnmatch
listing = os.scandir(image_dir)
for thisFile in listing:
    #if fnmatch.fnmatch(thisFile, '*936.jpg'):
    if fnmatch.fnmatch(thisFile, '*.jpg'):
        thisImgFile = image_dir + "/" + thisFile.name

        if device == "tpu":
            # Returns a numpy array: x1, x2, y1, y2, conf, class
            results = model.predict(thisImgFile, save_img=showInfResults, save_txt=showInfResults)
            inferTime = model.get_last_inference_time() #inference time, nms time
            print(f"Inference Time: {inferTime}")
            #if results.shape[0] != 1:
            #    print(results)
            print(f"Results: {type(results)}, {results}")


        else:
            # Returns a dict
            results = model.predict(thisImgFile)
            if debug:
                #print(f"Results: {type(results)}, {results}")
                #print(f"Results[0]: {type(results[0])}, {results[0]}")
                #print(f"Results shape: {type(results)}, {results.shape}")
                print(f"Results.boxes: {type(results[0].boxes.data)}, {results[0].boxes.data}")
                print(f"File: {thisImgFile}")

        print("---------------------------------------------")
        if device == "tpu":
            boxes = results
                #if results.shape[0] != 1: # If we have detected more than one object
                #if result[5] != 80:  # not a hand (47 is apple)
                    #print(f"result:{result}")
                    #logger.info("result:{}".format(result))
                    #validRes = distCalc.loadData(result)

        else:
            if showInfResults:
                print(results[0].boxes)
                results[0].show()

            boxes = results[0].boxes.data

            if debug:
                if device != "tpu":
                    #print(f"Results: {type(results[0].boxes.data)}, {results[0].boxes.data}")
                    print(f"Results: {type(results[0].boxes)}, {results[0].boxes}")

        validRes = distCalc.loadData(boxes, device)

        if debug:
            print(f"N objects detected: hands = {distCalc.nHands}, non hands = {distCalc.nNonHand}")
            print(f"Valid: {validRes}")

#            if validRes:
#                handObjDisp.draw(thisImgFile, distCalc)
