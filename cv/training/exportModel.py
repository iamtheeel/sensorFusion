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


def saveModel(modelFile, dataSet, imgH, imgW ):

    # Load the state dict
    #model.load_state_dict(torch.load(modelFile), strict=False) 
    model = YOLO(modelFile)  # build a new model from YAML


    '''
    must use python 3.11 for the exporter to work as of 7/7/24
    '''
    #model.export(format="tflite", data=dataSet, int8=True) 
    #model.export(format="tflite", data=dataSet, imgsz=(imgH, imgW), int8=True)
    # https://github.com/ultralytics/ultralytics/issues/1185  #I did not seem to have a problem tho
    model.export(format="edgetpu", data=dataSet, imgsz=(imgH, imgW), int8=True) #Edge TPU is linux only
    #Img is H, w
    #model.export(format="tflite", imgsz=(imgMax, imgMax), int8=True, data=dataSet, optimize=True) # 
    #model.export(format="saved_model", imgsz=(imgMax, imgMax) ) # save as a tensorflow model



#weightsDir = "runs/detect/train9/weights/" #glass: YoloV3, imgsz=320
#weightsDir = "runs/detect/train10/weights/" #glass: YoloV8, imgsz=320
#weightsDir = "runs/detect/train11/weights/" #glass: YoloV8, imgsz=96
#weightsDir = "runs/detect/train29/weights/" #glass: YoloV3, imgsz=320, d,w: 
#modelDir = "runs/detect/train30/weights/" 
modelDir = "../weights/" 
modelPath = Path(modelDir)
#fileName = "yolov5nu_orig.pt"
#fileName = "yolov5nu_transferFromCOCO.pt"
#fileName = "yolov5n_2class_ourDataTrained_320.pt"
#fileName = "yolov5nu_tran_hsv_h-1.0_2class.pt"
fileName = "yolov5nu_tran_hsv_h-1.0_81class.pt"
modelFile = modelPath/fileName

#dataSet = "coco8.yaml"
#dataSet = "datasets/coco8.yaml"
#dataSet = "datasets/dataset_ver1.yaml"
dataSet = "../datasets/coco_withHand.yaml" # 81 classes (coco + hand)
#dataSet = "datasets/combinedData.yaml" # 2 classes only
# img size should be multiple of 16
#imgH = 240
#imgH = 320
#imgH = 416 # 415 saved as 416
#imgH = 512
#imgH = 576
imgH = 608
#imgH = 624 # Failed
#imgH = 640 # seems to be borked
imgW = imgH
#imgW = 640

saveModel(modelFile=modelFile, dataSet=dataSet, imgH=imgH, imgW=imgW) 
