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
import yaml #To get names from class number

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("display")

#TODO: Change label position if the location obstructs it

class displayHandObject:
    def __init__(self,  config, camNum=1) -> None:
        logger.info("Init: displayHandObject")
        self.imageSize = config['training']['imageSize']
        self.frameRate = config['runTime']['camRateHz']
        conf = config['runTime']['displaySettings']
        self.fullScreen = conf['fullScreen']
        self.handLineTh = conf['handLineTh']
        self.objLineTh = conf['objLineTh']
        self.distLineTh = conf['distLineTh']
        self.videoFile = config['debugs']['videoFile']

        self.waitKeyTime = 0 #ms, wait until the key is pressed
        if(conf['runCamOnce'] == False):
            self.waitKeyTime = 1 #ms, will run through with a delay

        #cv2.namedWindow("sensorFusion", cv2.WINDOW_NORMAL )
        self.windowName = f"Camera {camNum}"
        cv2.namedWindow(self.windowName, cv2.WINDOW_AUTOSIZE )
        cv2.moveWindow(self.windowName, 10,10) #Does not work with Wayland
        #cv2.putText(img='', text="Loading", org=[10,10], fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=self.objectColor)

        ## Save video
        if self.videoFile != '':
            self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            logger.info(f"Saving video: {self.videoFile}, h, w: {self.imageSize[1]}, {self.imageSize[0]}, rate: {self.frameRate}")
            self.videoOut = cv2.VideoWriter(self.videoFile, fourcc, self.frameRate, (self.imageSize[1],self.imageSize[0]))



        # Save images
        self.saveFile = config['debugs']['saveImages']
        self.imageDir = config['runTime']['imageDir']
        if(not os.path.exists(self.imageDir)):
            logger.error(f"Image Dir does not exist, creating: {self.imageDir}")
            try:
                os.makedirs(self.imageDir)
            except Exception as e:
                print(f"Could not create {self.imageDir}, {e}")
                exit()

        yoloConfig = f"{config['training']['dataSetDir']}/{config['training']['dataSet']}"
        with open(yoloConfig, "r") as yoloConfigFile:
            yolo_config = yaml.safe_load(yoloConfigFile)
        self.class_names = yolo_config.get("names", []) # Get class names list
        # Define a list of distinct colors (BGR format for OpenCV)
        self.color_list = [
            (0, 255, 0),   # Apple: Green
            (0, 0, 128),   # Ball: Dark Red
            (0, 128, 128), # Bottle: Teal
            (0, 128, 0),   # Clip: Dark Green
            (0, 0, 0),     # Glove: black
            (255, 0, 0),   # Lid: Blue
            (255, 255, 0), # Plate: Cyan
            (128, 0, 128), # Spoon: Purple
            (255, 0, 255), # Tape Spool: Magenta
            (0, 0, 255),   # Dark Red
            (255, 255, 255)# Line: White
        ]

    def draw(self, thisImg, dist, valid, camNum=None, saveFileName="", asFile = False):
        if asFile:
            self.waitKeyTime = 0 #ms, will wait untill the key is pressed

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

        
        if self.videoFile != '': # Save video
            self.videoOut.write(thisImg)
        cv2.imshow(self.windowName, thisImg) # Display the image

        if(self.saveFile):
            startDateTime = '{date:%Y%m%d-%H%M%S-%f}'.format( date=datetime.datetime.now() )[:-3]
            saveFileName = f"{self.imageDir}/{camNum}_{startDateTime}_{saveFileName}.jpg"
            logger.info(f"Saving file: {saveFileName}")
            cv2.imwrite(saveFileName, thisImg)


        waitkey = cv2.waitKey(self.waitKeyTime)
        return waitkey

    def drawObject(self, thisImg, dist):
        # The Object
        detected_class_id = int(dist.grabObject[5])
        targetColor = self.color_list[detected_class_id % len(self.color_list)]
        objectname = self.class_names[detected_class_id] if detected_class_id < len(self.class_names) else "Unknown"

        objText = f"{objectname}: {dist.grabObject[4]:.2f}%"
        #logger.info(f"drawObject: {dist.grabObject}")
        objUL, objLR = dist.getBox(dist.grabObject)
        cv2.rectangle(img=thisImg, pt1=objUL, pt2=objLR, color=targetColor, thickness=self.objLineTh)
        cv2.circle(img=thisImg, center=dist.bestCenter, radius=5, color=targetColor, thickness=2) #BGR
        #cv2.putText(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.objectColor)
        self.putTextInBox(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, text_color=targetColor, box_color=[255, 255, 255])

    def drawHand(self, thisImg, dist):
        # The Hand
        handText = f"Glove: {dist.handObject[4]:.2f}%"
        handUL, handLR = dist.getBox(dist.handObject)
        cv2.rectangle(img=thisImg, pt1=handUL, pt2=handLR, color=self.color_list[4], thickness=self.handLineTh)
        cv2.circle(img=thisImg, center=dist.handCenter, radius=5, color=self.color_list[4], thickness=2) #BGR
        cv2.putText(img=thisImg, text=handText, org=handUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=self.color_list[4])

    def drawDistance(self, thisImg, dist):
        # The distance
        distText =f"Distance: {dist.bestDist:.0f}mm"
        #logger.info(distText)
        distUL =  (0,25)  # The location of the text box (x, y)
        cv2.line(img=thisImg, pt1=dist.bestCenter, pt2=dist.handCenter, color=self.color_list[10], thickness=self.distLineTh)
        #cv2.putText(img=thisImg, text=distText, org=distUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2, color=self.lineColor)
        self.putTextInBox(img=thisImg, text=distText, org=distUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=2, text_color=self.color_list[10], padding=5)

    def putTextInBox(self, img, text, org, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=1, text_color=(255, 255, 255), box_color=(0, 0, 0), padding=2):
        text_size, _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
        text_w, text_h = text_size  # Width and height of text

        # Calculate box coordinates
        box_x1, box_y1 = org[0] - padding, org[1] - text_h - padding
        box_x2, box_y2 = org[0] + text_w + padding, org[1] + padding

        # Ensure box is within image bounds
        box_x1, box_y1 = max(0, box_x1), max(0, box_y1)
        box_x2, box_y2 = min(img.shape[1], box_x2), min(img.shape[0], box_y2)

        # Draw filled rectangle (background for text)
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), box_color, cv2.FILLED)

        # Draw the text on top of the box
        cv2.putText(img, text, org, fontFace, fontScale, text_color, thickness)
