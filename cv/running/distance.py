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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Distance")

class distanceCalculator:
    def __init__(self, trainImgSize, config) -> None:
        ##Configs
        self.modelImgSize = trainImgSize #What is the pxl count of the image directly to inferance
        self.pxPerIn  = config['imagePxlPer_mm'] #How many pixels per mm
        self.hThresh = config['handThreshold']
        self.oThresh = config['objectThreshold']
        self.handClassNum = config['handClass']

        print(f"pxPerIn: {self.pxPerIn}")

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
        #logger.info(f"zeroData, Max dist: {self.bestDist}")

    def loadData(self, data, device):
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
        #self.device = device
        #if device == 'tpu': 
        classField = 5
        confField = 4

        self.zeroData()

        self.data = data
        
        logger.info(f"LoadData, Data: {data}")
        for object in data:
            #logger.info(f"LoadDataobject: {object}")
            #logger.info(f"LoadDatathis object class: {object[classField]}, hand class: {self.handClassNum}")
            if object[classField] == self.handClassNum and object[confField] >= self.hThresh:
                self.nHands += 1
                #self.hand = object
                self.handCenter = self.findCenter(object)
                self.handObject = object
            elif object[confField] >= self.oThresh: 
                logger.info(f"loadData, object: {object}")
                self.nNonHand += 1
                ## we still want to be able to display if we don't have a hand
                # use the best confidence untill we can be bothered to show multiple objectes
                if object[confField] > self.grabObject[confField]:
                    self.grabObject[confField] = object[confField]
                    self.grabObject = object # hmm, how to sort multiple objects if we have no hand?
                    self.bestCenter = self.findCenter(object)


        # If we have multiple hands use the one with the highest confidence
        if self.nHands >= 1:
            for object in data:
                if object[classField] == self.handClassNum and object[confField] >= self.hThresh and object[confField] > self.handConf:
                    self.handConf = object[confField]
                    self.handCenter = self.findCenter(object)
                    self.handObject = object

        #if len(data) < 2: 
        #    logger.info(f"loadData: must be more than 1 object. len of data: {len(data)}")
        #    return False

        if self.nNonHand == 0 or self.nHands == 0:
            logger.info(f"loadData: we need at least one hand:{self.nHands} and one target:{self.nNonHand}")
            return False
        
        # Once we have the hand object, get the closest distance
        for object in data:
            if object[classField] != self.handClassNum and object[confField] >= self.oThresh: 
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
