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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("display")

#TODO: Change label position if the location obstructs it

class displayHandObject:
    def __init__(self,  conf ) -> None:
        self.handColor = conf['handColor']
        self.objectColor = conf['objectColor']
        self.lineColor = conf['lineColor']
        self.fullScreen = conf['fullScreen']
        self.handLineTh = conf['handLineTh']
        self.objLineTh = conf['objLineTh']
        self.distLineTh = conf['distLineTh']
        self.conf = conf

        self.waitKeyTime = 0 #ms, wait until the key is pressed
        if(self.conf['runCamOnce'] == False):
            self.waitKeyTime = 50 #ms, will run through with a delay
        #logger.info(f"waitKeyTime: {self.waitKeyTime}")

    def draw(self, imgFile, dist, valid):
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

        cv2.namedWindow("sensorFusion", cv2.WINDOW_NORMAL )
        if self.fullScreen:
            cv2.setWindowProperty("sensorFusion",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN) 
            # prevents cv2 from showing on the corel board, works on MAC, on linux has full screen - but image is not

        #cv2.setWindowProperty("sensorFusion",cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("sensorFusion", thisImg)

        waitkey = cv2.waitKey(self.waitKeyTime)
        return waitkey

    def drawObject(self, thisImg, dist):
        # The Object
        objText = f"Target[{dist.grabObject[5]:.0f}]: {dist.grabObject[4]:.2f}"
        logger.info(f"drawObject: {dist.grabObject}")
        objUL, objLR = dist.getBox(dist.grabObject)
        cv2.rectangle(img=thisImg, pt1=objUL, pt2=objLR, color=self.objectColor, thickness=self.objLineTh)
        cv2.circle(img=thisImg, center=dist.bestCenter, radius=5, color=self.objectColor, thickness=2) #BGR
        cv2.putText(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.objectColor)

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
        distUL =  (5,15)
        logger.info(distText)
        cv2.line(img=thisImg, pt1=dist.bestCenter, pt2=dist.handCenter, color=self.lineColor, thickness=self.distLineTh)
        cv2.putText(img=thisImg, text=distText, org=distUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.lineColor)
