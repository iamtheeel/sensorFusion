import os, fnmatch
import numpy as np
import cv2

import torchvision.transforms as trans

from getLab import getLab


# Settings
modelType = 'ssd' # is this yolo or ssd?
#modelType = 'yolo' # is this yolo or ssd?

#thisFile = None
thisFile = 'dataset3_0.jpg'

if modelType == 'ssd':
    dataSet = 'dataset_ver1'
    testTrain = 'test' # train or test
    baseDir = "../../data/version1" 
    image_dir = baseDir + '/' + dataSet + '/' + testTrain
else:
    dataSet = 'coco8'
    testTrain = 'val' # train or val
    baseDir = "datasets" 
    image_dir = baseDir + '/' + dataSet + '/images/' + testTrain

print(f"{image_dir}")

def getThisImgLabs(image_loc, fileName):
    print(f"getThisLab file: {fileName}")

    imageFile_str =image_loc+'/'+fileName
    image = cv2.imread(imageFile_str) 
    nLab = labeler.getNLab(imageFile_str) # Sets up file structure for this file, and returns the number of lables
    print(f"getThisLab num labels: {nLab}")

    # Add the labels and boxes to the image
    for thisLab in range(nLab):
        label, UL, LR = labeler.getLabBox(thisLab)
        print(f"label: {thisLab}, {label}")
        cv2.rectangle(img=image, pt1=UL, pt2=LR, color=(0,255,0), thickness=1)
        cv2.putText(img=image, text=label, org=UL, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))

    # Display the image
    numpy_arr = np.asarray(image)
    cv2.imshow(fileName, numpy_arr)
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