#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Display the image with lable(s) and box(s)
# Either single image or all the images in a dir.
#   for image dir thisFile = None
#
###


import os, fnmatch
import numpy as np
import cv2

import torchvision.transforms as trans

from getLab import getLab


# Settings
#modelType = 'ssd' # is this yolo or ssd?
modelType = 'yolo' # is this yolo or ssd?

#thisFile = None
#thisFile = 'Dataset_apple_97.jpg'
thisFile = 'dataset_ver1_dataset8_958.jpg'
#thisFile = 'dataset8_958.jpg'

if modelType == 'ssd':
    dataSet = 'dataset_ver1'
    testTrain = 'train' # train or test
    baseDir = "../../data/version1" 
    image_dir = baseDir + '/' + dataSet + '/' + testTrain
else:
    #dataSet = 'coco8'
    #dataSet = 'dataset_ver1'
    dataSet = 'combinedData'
    testTrain = 'val' # train or val
    baseDir = "datasets" 
    image_dir = baseDir + '/' + dataSet + '/images/' + testTrain

print(f"{image_dir}")

def labelColor(label):
    # color: (b, g, r)
    match label:
        case 'hand': return (255, 255, 255) # White
        case 'apple': return (0, 0, 155)
        case _: return (0,0,0) # Black

def getThisImgLabs(image_loc, fileName):
    print(f"getThisLab file: {fileName}")

    imageFile_str =image_loc+'/'+fileName
    image = cv2.imread(imageFile_str) 
    nLab = labeler.getNLab(imageFile_str) # Sets up file structure for this file, and returns the number of lables
    print(f"getThisLab num labels: {nLab}")

    # Add the labels and boxes to the image
    for thisLab in range(nLab):
        label, UL, LR = labeler.getLabBox(thisLab)
        color = labelColor(label)
        print(f"label: {thisLab+1}, {label}")
        cv2.rectangle(img=image, pt1=UL, pt2=LR, color=color, thickness=1)
        cv2.putText(img=image, text=label, org=UL, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color)

    # Display the image
    # TODO: add exit key stroke
    numpy_arr = np.asarray(image)
    cv2.imshow('', numpy_arr) # if you name with the image, will make a new window each time
    cv2.waitKey()

# Entry point
labeler = getLab(modelType, baseDir, dataSet)  

if thisFile != None:
    getThisImgLabs(image_dir, thisFile)

else:
    listing = os.scandir(image_dir)
    for file in listing:
        if fnmatch.fnmatch(file, '*.jpg'):
            getThisImgLabs(image_dir, file.name)