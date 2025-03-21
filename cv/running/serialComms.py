#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Spring 2025
#
###
#
# Serial Comms to the ExoGlove
#
###

#https://docs.google.com/document/d/1g40Lloi58_x_gf3x7uav3_gDUWDt-gu5-2V8a9qaB2c/edit?tab=t.0#heading=h.9awg72r7te06
# $CV, imgCapTime (uint_32), handConf (uint8), object class (uint8), object conf (uint8), Distance (uint16), <LF><CR>

# Connect at 115200-8N1
# Serial Comms at 5V TTY
import serial  #pip install pyserial

## Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class commsClass:
    def __init__(self, config):
        self.port = config['port']      #: "asdf"
        self.speed = config['speed']     #: 115200
        self.dataBits = config['dataBits']  #: 8
        self.parity = config['parity']    #: N
        self.stopBits = config['stobBits']   #: 1
        self.id = config['id']   

        #Serial<id=0xa81c10, open=False>(port='COM1', baudrate=19200, bytesize=8, parity='N', stopbits=1, timeout=None, xonxoff=0, rtscts=0)
        if self.port!= "None":
            self.ser = serial.Serial(port=self.port, baudrate=self.speed, bytesize=self.dataBits, parity=self.parity, stopbits=self.stopBits)

    def close(self):
        self.ser.close()


    def sendString(self, timeMS=0, handConf=0, object=None, objectConf=0, distance=0):
        sendStr = f"${self.id},{timeMS},{handConf:.2f},{int(object)},{objectConf:.2f},{distance:.0f}\n\r" #\n\r = <LF> <CR>: 0x0A 0x0D
        if self.port!= "None":
            self.ser.write(sendStr.encode('utf-8'))  # Encode as 8 bit bytes
            logger.info(f"Sending: {sendStr}")

        pass
