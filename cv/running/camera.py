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

class camera:
    def __init__(self, config):
        self.config = config
        self.camStat = None
        self.image = None
        self.thisCam = None

        ## Logging
        debug = config['debugs']['debug']
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if debug == False:
            logging.disable(level=logging.CRITICAL)
            self.logger.disabled = True

        camID = config['runTime']['camId']

        # right now there are two options, USB id (int), or rtsp (string)
        self.camType = 'USB'
        if(type(camID) == str): self.camType = 'rtsp'
        
        self.logger.info(f"camID: {self.camType} Type: {type(camID)}")

        if(self.camType == 'rtsp'):
            # https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'#;appsink|sync;false'
        else:
            # the rtsp can not change settings
            camera.set(cv2.CAP_PROP_FPS, config['runTime']['camRateHz'])
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  config['training']['imageSize'][0])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['training']['imageSize'][1])

        self.startStream()
        self.logger.info(f"Camera Stream Started")
        self.camTimeout_ns = 50*1000*1000 # 30 ms

    def startStream(self):
        # TODO: check to see if camera is there
        self.thisCam = cv2.VideoCapture(self.config['runTime']['camId'], cv2.CAP_ANY)

    def getImage(self):
        if self.camStat and self.camType == 'rtsp':
            self.image = cv2.resize(self.image, self.config['training']['imageSize'])
        return self.camStat, self.image

    def capImage (self):

        camReadTime_ms = 0
        while(camReadTime_ms < 10): #if our grab time is < 30 milisec we are behind
          # VideoCapture::waitAny() is supported by V4L backend only
          #if self.thisCam.waitAny([self.thisCam],  self.camTimeout_ns): # wait for produce a frame
          self.logger.info(f"Get the next frame")
          tStart = time.time_ns()
          self.camStat, self.image = self.thisCam.read()
          #self.thisCam.grab()
          camReadTime_ms = (time.time_ns()-tStart)/(1e6)
          self.logger.info(f"grab status: {self.camStat}, {camReadTime_ms:.3f}ms")

          #else:
          #  # We seem to have lost our camera, try restarting
          #  self.logger.error(f"Camera Seems Borked, restart stream")
          #  self.startStream()
