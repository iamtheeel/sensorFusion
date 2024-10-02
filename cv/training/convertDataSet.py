#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Convert the dataset to the YOLO format
#
# Input format (xml): 
#   label = text, 
#   box = pt1(x, y), pt2(x, y): Number of pixles
#
# Output format (txt): 
#   lebel = int 
#   Box = x, y, w, h: percent of image
###

import os, fnmatch, cv2
import yaml
import random

from getLab import getLab

## Settings
# Inputs:
baseDir = "../../data/version1" 

#dataSetList = ['appleHand'] # For testing
dataSetList = ['Dataset', 'dataset_ver1', 'dataset_ver2', 'dataset_ver3_white_background']

# Outputs
#outName = 'foo' # small set for testing
#outName = 'combinedData' # 2 class
outName = 'coco_8Class' # coco pluss hand
outBaseDir = "./datasets"


outPath = outBaseDir+'/'+outName
outConfigFile = outName + ".yaml"
trainPerc = 90

# Parse the output ymal
with open(outBaseDir+'/'+outConfigFile, 'r') as confFile:
    config = yaml.load(confFile, Loader=yaml.SafeLoader)

trainSplit = config['train'].split("/")
testSplit = config['val'].split("/")
trainLabDir = (os.path.join(outPath, 'labels', trainSplit[1])) #TODO, this will break if we are not 2 deep
testLabDir = (os.path.join(outPath, 'labels', testSplit[1]))    #TODO, but we are, so there for now

print(f"{trainLabDir}")
trainDir = os.path.join(outPath, config['train'])
testDir  = os.path.join(outPath, config['val'])
#print(f"output path: {config['path']}")
print(f"output path: {outPath}")
print(f"tain dir: {trainDir}")
print(f"test dir: {testDir}")
print(f"names count: {len(config['names'])}")
#print(f"names: {config['names'][0]}")

# make the directories we will need
#if testTrain == 'train':
os.makedirs(trainDir, exist_ok=True)
os.makedirs(trainLabDir, exist_ok=True)
#else:
os.makedirs(testDir, exist_ok=True)
os.makedirs(testLabDir, exist_ok=True)


for dataSet in dataSetList:
    if dataSet == 'dataset_ver3_white_background' or dataSet == 'appleHand':
        subDirList = ['.'] # no sub dir
    elif dataSet == 'Dataset':
        #subDirList = ['apple', 'hand']
        subDirList = ['apple', 'hand', 'banana', 'can', 'marker', 'toothpaste']
    else: 
        subDirList = ['test', 'train']

    for subDir in subDirList:
        image_dir = baseDir + '/' + dataSet + '/' + subDir
        print(f"dir: {image_dir}")

        labeler = getLab('ssd', baseDir, dataSet)  

        listing = os.scandir(image_dir)
        for file in listing:
            if fnmatch.fnmatch(file, '*.jpg'):
                newFileName = dataSet + '_' + file.name # We have duplicate names
                imageFile_str =image_dir+'/'+file.name

                # Sets up file structure for this file, and returns the number of lables
                nLab = labeler.getNLab(imageFile_str) 

                #Get the image dimentions
                img = cv2.imread(imageFile_str)
                imgH, imgW, imgC = img.shape

                # Get the name of the label file 
                #fileSplit = file.name.split('.')
                fileSplit = newFileName.split('.')
                labelFile = fileSplit[0]+".txt"
                #print(f"image file: {file.name}, label file: {labelFile}")

                # split to test/train 
                testTrainVal = random.randint(0, 100)
                if testTrainVal < trainPerc:
                    imgDir = trainDir
                    labDir = trainLabDir
                else: 
                    imgDir = testDir
                    labDir = testLabDir

                #print(f"file: {file.name}, testTrain: {testTrainVal}")
                # Don't write if we don't have a lable
                if nLab > 0:
                    os.system(f'cp {imageFile_str} {imgDir}/{newFileName}')
                    labFileWriter = open(labDir+'/'+labelFile, 'w')

                    #print(f"nLab: {nLab}, w: {imgW}, h: {imgH}")

                    for thisLab in range(nLab):
                        # Get the box info 
                        label, UL, LR = labeler.getLabBox(thisLab, asInt=False)

                        # Get the label number
                        for key, item in config['names'].items():
                            #print(f"key: {key}, item: {item}")
                            if item == label:
                                break
                        #print(f"key: {key}, item: {item}")

                        #print(f"label: {label}, pt1(x, y): ({UL[0]}, {UL[1]}), pt2(x, y): ({LR[0]}, {LR[1]})")
                        x = ((UL[0] + LR[0])/2) / imgW
                        y = ((UL[1] + LR[1])/2) / imgH
                        w = (LR[0] - UL[0]) / imgW
                        h = (LR[1] - UL[1]) / imgH
                        labFileWriter.write(f"{key} {x} {y} {w} {h}\n")
                        #print(f"    key: {key}, loc(x, y): ({x}, {y}), size(w, h): ({w}, {h})")