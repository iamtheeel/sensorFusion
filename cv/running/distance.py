#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Calculate the distance between two items
#
###

#TODO: improve data structure

import math
import torch

class distanceCalculator:
    def __init__(self, trainImgSize, pxPerInCal, handThresh =0.0, objThresh=0.0) -> None:
        ##Configs
        self.modelImgSize = trainImgSize #What is the pxl count of the image directly to inferance
        self.pxPerIn  = pxPerInCal #How many pixels per mm
        self.hThresh = handThresh
        self.oThresh = objThresh

        self.zeroData()

    def zeroData(self):
        self.nHands = 0
        self.nNonHand = 0
        self.grabObject = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) # Init to UL
        self.handObject = None

        self.bestCenter = (0,0)
        self.handCenter = (self.modelImgSize[0], self.modelImgSize[1])            # Init to LR
        self.handConf = 0.0

        self.bestDist = self.calcDist(self.grabObject)
        #print(f"Max dist: {self.bestDist}")

    def loadData(self, data, cls):
        '''
        Return True iff there is one and only one hand
        Loads the data used
        Converts to actual image size
        Args:
            (tenor data): x, y, w, h, probability, class for each item
            (list of classes seen): 
        Returns:
            (Bool): Is or is not valid
        Raises:
        '''
        self.zeroData()

        self.data = data
        if len(data) < 2: 
            print(f"loadData: must be more than 1 object. len of data: {len(data)}")
            return False
        
        for object in data:
            if object[5] == 0 and object[4] >= self.hThresh:
                self.nHands += 1
                #self.hand = object
                self.handCenter = self.findCenter(object)
                self.handObject = object
            elif object[4] >= self.oThresh: 
                self.nNonHand += 1

        if self.nNonHand == 0 or self.nHands == 0:
            print(f"loadData: we need at least one hand:{self.nHands} and one target:{self.nNonHand}")
            return False

        # If we have multiple hands use the one with the highest confidence
        if self.nHands >= 1:
            for object in data:
                if object[5] == 0 and object[4] >= self.hThresh and object[4] > self.handConf:
                    self.handConf = object[4]
                    self.handCenter = self.findCenter(object)
                    self.handObject = object
        
        # Once we have the hand object, get the closest distance
        for object in data:
            if object[5] != 0 and object[4] >= self.oThresh: 
                thisDist = self.calcDist(object)
                if thisDist < self.bestDist: 
                    self.grabObject = object
                    self.bestDist = thisDist
                    self.bestCenter = self.findCenter(object)

        return True

    def calcDist(self, object):
        '''
        Calculates the distance between the hand and the object
        Args:
        Returns:
            (float): The distance in mm
        Raises:
        '''
        center = self.findCenter(object)
        yDist = (self.handCenter[0]- center[0])
        xDist = (self.handCenter[1]- center[1])

        pxlDist = math.sqrt(pow(yDist,2) + pow(xDist,2))
        return pxlDist/self.pxPerIn

    def getBox(self, object):
        '''
        Gets the object box
        Args:
            (tenor data): x1, y1, x2, y2, probability, class for each item
        Returns:
            (Upper left and Lower Right Cornersf of the box)

        '''
        x1, y1, x2, y2 = self.getXY(object)
        UL = [int(x1), int(y1)]
        LR = [int(x2), int(y2)]

        return UL, LR

    def findCenter(self, object):
        '''
        Gets the center of an object
        Args:
            (tenor data): x, y, w, h, probability, class for each item
        Returns:
            (int x, int y): the center of the object in pxles
        Raises:
        '''

        '''
        # corner/size
        #from https://github.com/MIC-Laboratory/Prosthetic_hand_control_MQTT_SSDMobileNet/blob/master/openMV/distance_sender_auto_exposure.py
        center_x = math.floor(x + (w / 2))
        center_y = math.floor(y + (h / 2))
        '''
        #Corner/corner
        x1, y1, x2, y2 = self.getXY(object)
        center_x = math.floor(x1 + (x2 - x1)/2)
        center_y = math.floor(y1 + (y2 - y1)/2)

        return center_x, center_y
    
    def getXY(self, object):
        x1 = object[0].item()
        y1 = object[1].item()
        x2 = object[2].item()
        y2 = object[3].item()

        return x1, y1, x2, y2