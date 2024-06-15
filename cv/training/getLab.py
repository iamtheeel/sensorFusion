#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Returns the labels and box dimensions for a single image
#   label as:
#       text
#   box dim as:
#        upper right (x, y)
#        lower left(x, y)
#
###

import os
import yaml
import cv2
import xml.etree.ElementTree as ET

class getLab:
    def __init__(self, type, baseDir, dataSet) -> None:
        self.type = type
        self.baseDir = baseDir

        self.labFile = ''
        self.imgFile = ''

        #self.xmlTree = None 
        self.xmlRoot = None

        print(f"type: {type}")
        if type == 'ssd':
            # class name in label file
            self.ext = 'xml'
        else:
            configFile = baseDir + '/' + dataSet + '.yaml'
            with open(configFile, 'r') as configStr:
                config = yaml.safe_load(configStr)
            self.classes = config["names"]
            print(f"N Classes: {len(self.classes)}")
            thisClass = 20
            print(f"Class: {thisClass}, {self.classes[thisClass]}")
            self.ext = 'txt'

    def getNLab(self, imgFile_str):
        self.imgFile = imgFile_str
        head_tail = os.path.split(self.imgFile)
        #print(f"Path: {head_tail[0]}, file = {head_tail[1]}")

        file_ext = os.path.splitext(head_tail[1])
        #print(f"fileName: {file_ext[0]}, extension = {file_ext[1]}")
        fileExt = file_ext[0] + '.' + self.ext

        if self.type == 'ssd':
            nLab = 0 

            labDir = head_tail[0]
            print(f"labDir: {labDir}")
            self.labFile = labDir + '/' + '/'+ fileExt
            print(f"Label file: {self.labFile}")

            xmlTree = ET.parse(self.labFile)
            self.xmlRoot = xmlTree.getroot()
            #for child in self.xmlRoot:
            #     print(f"tag: {child.tag}, attrib: {child.attrib}")
            #     if child.tag == 'object':
            #        nLab += 1
            for object in self.xmlRoot.findall('object'):
                nLab += 1

        else:
            train_val = os.path.split(head_tail[0])
            lab_dir = os.path.split(train_val[0]) 
            labDir = lab_dir[0]+ '/labels'
            self.labFile = labDir + '/' + train_val[1] + '/'+ fileExt

            print(f"Label file: {self.labFile}")
            with open(self.labFile, 'r') as fp: 
                nLab = len(fp.readlines())

        return nLab

    
    def getLabBox(self, labNum):
        if self.type == 'ssd':
            label, UL, LR = self.getLabBoxSSD(labNum)
        else:
            img = cv2.imread(self.imgFile)
            imgH, imgW, imgC = img.shape
            label, UL, LR = self.getLabBoxYOLO(labNum, imgH, imgW, imgC)

        return label, UL, LR

    def getLabBoxSSD(self, labNum):
        thisLab = 0
        #print(f"getLabBoxSSD labNum: {labNum}")
        for object in self.xmlRoot.findall('object'):
            if thisLab == labNum:
                name = object.find('name').text
                #print(f"getLabBoxSSD label: {name}")

                bndBox = object.find('bndbox')
                xmin = int(float(bndBox.find('xmin').text))
                xmax = int(float(bndBox.find('xmax').text))
                ymin = int(float(bndBox.find('ymin').text))
                ymax = int(float(bndBox.find('ymax').text))
                #print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")
                return name, (xmin,ymin), (xmax,ymax)
            thisLab += 1

    def getLabBoxYOLO(self, labNum, imgH, imgW, imgC):
        # label, x, y, w, h
        #print(f"h: {imgH}, w: {imgW}, ch: = {imgC}")
        with open(self.labFile, 'r') as fp: 
            lines = fp.readlines()

        #print(f"getLabBox: line = {lines[labNum]}")
        line_split = lines[labNum].split(" ")
        labelNum = int(line_split[0])
        X = float(line_split[1]) * imgW
        Y = float(line_split[2]) * imgH
        w = float(line_split[3]) * imgW
        h = float(line_split[4]) * imgH

        # Moves to get lab box?
        upperLeft_X = int(X - w/2)
        upperLeft_Y = int(Y - h/2)
        lowerRight_X = int(X + w/2)
        lowerRight_Y = int(Y + h/2)

        UL = [upperLeft_X, upperLeft_Y]
        LR = [lowerRight_X, lowerRight_Y]

        #print(f"label number: {line_split[0]}, {self.classes[labelNum]}")
        return self.classes[labelNum], UL, LR