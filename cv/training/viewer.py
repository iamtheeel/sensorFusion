import os, fnmatch
import numpy as np
import cv2

import torchvision.transforms as trans

from getLab import getLab


dataSet = 'coco8'
testTrain = 'val' # train or val

# is this yolo or CNN?

# make a read label function
# getNumLabels
# getLabel(n)

baseDir = "datasets" 
image_dir = baseDir + '/' + dataSet + '/images/' + testTrain

labeler = getLab('yolo', baseDir, dataSet) # ssd, or yolo

print(f"{image_dir}")
listing = os.scandir(image_dir)
for file in listing:
    if fnmatch.fnmatch(file, '*.jpg'):
        imageFile_str =image_dir+'/'+file.name
        print(f"{imageFile_str}")
        #image = Image.open(imageFile_str, ) 
        image = cv2.imread(imageFile_str) 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        nLab = labeler.getNLab(imageFile_str) # Sets up file structure for this file, and returns the number of lables
        for thisLab in range(nLab):
            label, UL, LR = labeler.getLabBox(thisLab)
            print(f"label: {thisLab}, {label}")
            cv2.rectangle(img=image, pt1=UL, pt2=LR, color=(0,255,0), thickness=1)
            cv2.putText(img=image, text=label, org=UL, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))
        #exit()

        numpy_arr = np.asarray(image)
        cv2.imshow('', numpy_arr)
        cv2.waitKey()