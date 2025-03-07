#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# View the results
#
###
import cv2
import datetime
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("display")

#TODO: Change label position if the location obstructs it

class displayHandObject:
    def __init__(self,  config, camNum=1) -> None:
        logger.info("Init: displayHandObject")
        conf = config['runTime']['displaySettings']
        self.handColor = conf['handColor']
        self.objectColor = conf['objectColor']
        self.lineColor = conf['lineColor']
        self.fullScreen = conf['fullScreen']
        self.handLineTh = conf['handLineTh']
        self.objLineTh = conf['objLineTh']
        self.distLineTh = conf['distLineTh']

        self.waitKeyTime = 0 #ms, wait until the key is pressed
        if(conf['runCamOnce'] == False):
            self.waitKeyTime = 1 #ms, will run through with a delay

        #cv2.namedWindow("sensorFusion", cv2.WINDOW_NORMAL )
        self.windowName = f"Camera {camNum}"
        cv2.namedWindow(self.windowName, cv2.WINDOW_AUTOSIZE )
        cv2.moveWindow(self.windowName, 10,10) #Does not work with Wayland
        #cv2.putText(img='', text="Loading", org=[10,10], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=self.objectColor)
        self.saveFile = config['debugs']['saveImages']
        self.imageDir = config['runTime']['imageDir']
        if(not os.path.exists(self.imageDir)):
            logger.error(f"Image Dir does not exist, creating: {self.imageDir}")
            try:
                os.makedirs(self.imageDir)
            except Exception as e:
                print(f"Could not create {self.imageDir}, {e}")
                exit()

    def draw(self, imgFile, dist, valid, camNum, saveFileName=""):
        if isinstance(imgFile, str):
            thisImg =  cv2.imread(imgFile)
            #logger.info(f"Image File: {imgFile}")

            self.waitKeyTime = 0 #ms, will wait untill the key is pressed
        else:
            thisImg =  imgFile

        #logger.info(f"Image File shape: {thisImg.shape}")

        if dist.nHands != 0:
            self.drawHand(thisImg, dist)

        if dist.nNonHand != 0:
            self.drawObject(thisImg, dist)

        if valid:
            self.drawDistance(thisImg, dist)

        if self.fullScreen:
            cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN) 
            # linux has full screen - but image is not
        #else:
            #cv2.setWindowProperty(self.windowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_AUTOSIZE) 

        #cv2.setWindowProperty(self.windowName,cv2.WND_PROP_TOPMOST, 1)

        cv2.imshow(self.windowName, thisImg)

        if(self.saveFile):
            startDateTime = '{date:%Y%m%d-%H%M%S-%f}'.format( date=datetime.datetime.now() )[:-3]
            saveFileName = f"{self.imageDir}/{camNum}_{startDateTime}_{saveFileName}.jpg"
            logger.info(f"Saving file: {saveFileName}")
            cv2.imwrite(saveFileName, thisImg)


        waitkey = cv2.waitKey(self.waitKeyTime)
        return waitkey

    def drawObject(self, thisImg, dist):
        # The Object
        objText = f"Target[{dist.grabObject[5]:.0f}]: {dist.grabObject[4]:.2f}"
        #logger.info(f"drawObject: {dist.grabObject}")
        objUL, objLR = dist.getBox(dist.grabObject)
        cv2.rectangle(img=thisImg, pt1=objUL, pt2=objLR, color=self.objectColor, thickness=self.objLineTh)
        cv2.circle(img=thisImg, center=dist.bestCenter, radius=5, color=self.objectColor, thickness=2) #BGR
        cv2.putText(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.objectColor)
        #cv2.putText(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.objectColor)

    def drawHand(self, thisImg, dist):
        # The Hand
        handText = f"Hand: {dist.handObject[4]:.2f}"
        handUL, handLR = dist.getBox(dist.handObject)
        cv2.rectangle(img=thisImg, pt1=handUL, pt2=handLR, color=self.handColor, thickness=self.handLineTh)
        cv2.circle(img=thisImg, center=dist.handCenter, radius=5, color=self.handColor, thickness=2) #BGR
        cv2.putText(img=thisImg, text=handText, org=handUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=self.handColor)

    def drawDistance(self, thisImg, dist):
        # The distance
        distText =f"Distance: {dist.bestDist:.0f}mm"
        distUL =  (5,40)  # The location of the text box
        logger.info(distText)
        cv2.line(img=thisImg, pt1=dist.bestCenter, pt2=dist.handCenter, color=self.lineColor, thickness=self.distLineTh)
        cv2.putText(img=thisImg, text=distText, org=distUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2, color=self.lineColor)
