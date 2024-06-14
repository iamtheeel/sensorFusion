

import os
import yaml
import cv2

class getLab:
    def __init__(self, type, baseDir, dataSet) -> None:
        self.type = type
        self.baseDir = baseDir

        self.labFile = ''
        self.imgFile = ''

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
            labDir = head_tail[0]
        else:
            train_val = os.path.split(head_tail[0])
            lab_dir = os.path.split(train_val[0]) 
            labDir = lab_dir[0]+ '/labels'

        self.labFile = labDir + '/' + train_val[1] + '/'+ fileExt
        #print(f"Label file: {self.labFile}")

        with open(self.labFile, 'r') as fp: 
            nLab = len(fp.readlines())

        return nLab
    
    def getLabBox(self, labNum):
        with open(self.labFile, 'r') as fp: 
            img = cv2.imread(self.imgFile)
            imgH, imgW, imgC = img.shape
            #print(f"h: {imgH}, w: {imgW}, ch: = {imgC}")
            lines = fp.readlines()
            #print(f"getLabBox: line = {lines[labNum]}")
            line_split = lines[labNum].split(" ")
            labelNum = int(line_split[0])
            X = float(line_split[1]) * imgW
            Y = float(line_split[2]) * imgH
            w = float(line_split[3]) * imgW
            h = float(line_split[4]) * imgH

            upperLeft_X = int(X - w/2)
            upperLeft_Y = int(Y - h/2)
            lowerRight_X = int(X + w/2)
            lowerRight_Y = int(Y + h/2)

            UL = [upperLeft_X, upperLeft_Y]
            LR = [lowerRight_X, lowerRight_Y]

            #print(f"label number: {line_split[0]}, {self.classes[labelNum]}")
            return self.classes[labelNum], UL, LR