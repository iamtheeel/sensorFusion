#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Camera
#
###

import logging
import os
import cv2
import time
import numpy as np

class camera:
    def __init__(self, config, camID):
        self.config = config
        self.camStat = None
        self.image = None
        self.thisCam = None
        self.imgH = config['training']['imageSize'][0]
        self.imgW = config['training']['imageSize'][1]

        ## Logging
        debug = config['debugs']['debug']
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if debug == False:
            logging.disable(level=logging.CRITICAL)
            self.logger.disabled = True

        self.camID = camID
        #self.camID = config['runTime']['camId']

        # right now there are two options, USB id (int), or rtsp (string)
        self.camType = 'USB'
        if(type(camID) == str): self.camType = 'rtsp'
        
        self.logger.info(f"camID: {self.camID} Type: {self.camType}")

        self.startStream()
        self.logger.info(f"Camera Stream Started")
        self.camTimeout_ns = 50*1000*1000 # 30 ms

        if(self.camType == 'rtsp'):
            # https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'#;appsink|sync;false'

        self.setZeroTime()
        #Preload the cap time in case we request an image that is not there
        capTime_ms = (time.time()*1000)
        self.thisCapTime = np.uint32(capTime_ms - self.zeroTime_ms)


    def __del__(self):
        self.logger.info(f"Closing Camera Stream")
        self.thisCam.release()

    def startStream(self):
        # TODO: check to see if camera is there
        self.thisCam = cv2.VideoCapture(self.camID, cv2.CAP_ANY)
        if(self.camType != 'rtsp'):
            # the rtsp can not change settings
            print(f"Cam settings")
            self.thisCam.set(cv2.CAP_PROP_FPS, self.config['runTime']['camRateHz'])
            if(self.config['runTime']['focus'] > -1):
                self.thisCam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.thisCam.set(cv2.CAP_PROP_FOCUS, self.config['runTime']['focus'])
            #self.thisCam.set(cv2.CAP_PROP_FOCUS, 10)
            self.thisCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgH)
            self.thisCam.set(cv2.CAP_PROP_FRAME_WIDTH,  self.imgW)

    def getImage(self):
        if self.camStat and self.camType == 'rtsp':
            # Camera is set to WVGA(480x848): labeled as WVGA, but it is FWVGA
            #Crop the image to what we are expecting
            self.logger.info(f"Image size (h, w, ch): {self.image.shape}, correctW: {self.imgW}")
            imgH, imgW, _ = self.image.shape
            if imgW > self.imgW:
                newWidth = int(self.imgW*(3/4)) # Fudge to change the aspect ratio
                cropCenter = int((imgW - newWidth)/2) 
                self.image = self.image[0:self.imgH, cropCenter:cropCenter+newWidth]

            #self.logger.info(f"Image new size (h, w, ch): {self.image.shape}")
            # Resize changes the aspect ratio
            self.image = cv2.resize(self.image, (self.imgW, self.imgH))

        return self.camStat, self.image, self.thisCapTime.copy()
    
    def setZeroTime(self):
        self.zeroTime_ms = (time.time()*1000)

    def capImage (self):

        capTime_ms = (time.time()*1000)

        if self.camStat and self.camType == 'rtsp':
            # VideoCapture::waitAny() is supported by V4L backend only
            #if self.thisCam.waitAny([self.thisCam],  self.camTimeout_ns): # wait for produce a frame
            #self.logger.info(f"Get the next frame")
            self.camStat, self.image = self.thisCam.read()
            '''
            camReadTime_ms = (time.time_ns()-tStart)/(1e6)
            there was a while to run this untill the Read time is > a threshold
            Hoever we don't need it anymore as we are running through a thread
            #self.logger.info(f"grab status: {self.camStat}, {camReadTime_ms:.3f}ms")
            '''
        
            if self.camStat == False:
                self.logger.error(f"Camera Seems Borked, restart stream")
                del self.thisCam
                self.startStream

        else:
            self.camStat, self.image = self.thisCam.read()

        # Get the time after the read, as the read waits for a new image
        self.thisCapTime = np.uint32(capTime_ms - self.zeroTime_ms)
        #self.logger.info(f"Zero time: {self.zeroTime_ms}ms, Cam read time: {type(capTime_ms)}, {capTime_ms}ms, zeroedCapTime: {type(self.thisCapTime)} {self.thisCapTime}ms")
