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
        #self.configs = config

        ## Logging
        debug = config['debugs']['debug']
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        if debug == False:
            logging.disable(level=logging.CRITICAL)
            self.logger.disabled = True

        camID = config['runTime']['camId']

        # right now there are two options, USB id (int), or rtsp (string)
        camType = 'USB'
        if(type(camID) == str):
            camType = 'rtsp'
        
        self.logger.info(f"camID: {camType} Type: {type(camID)}")

        if(camType == 'rtsp'):
            # https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'#;appsink|sync;false'
        else:
            # the rtsp can not change settings
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['training']['imageSize'][0])
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,  config['training']['imageSize'][1])

        self.thisCam = cv2.VideoCapture(config['runTime']['camId'], cv2.CAP_ANY)

        # TODO: check to see if camera is there

    def getImage (self):
        tStart = time.time_ns()
        camStat, image = self.thisCam.read()
        #self.thisCam.grab()
        camReadTime_ms = (time.time_ns()-tStart)/(1e6)
        self.logger.info(f"grab status: {camStat}, {camReadTime_ms:.3f}ms")

        while(camReadTime_ms < 10): #if our grab time is < 30 milisec we are behind
            tStart = time.time_ns()
            camStat, image = self.thisCam.read()
            #self.thisCam.grab()
            camReadTime_ms = (time.time_ns()-tStart)/(1e6)
            self.logger.info(f"grab status: {camStat}, {camReadTime_ms:.3f}ms")

        return camStat, image