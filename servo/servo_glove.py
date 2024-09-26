# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Fall 2024
#
###
#
# Example demonstrating the use of servo_I2C
# Using a PCA9685 16CH 12Bit PWM Led Driver
# 
###

import servo_I2C

import sys
import os
from time import sleep

# From MICLab
sys.path.insert(0, '..')
from ConfigParser import ConfigParser

## Configuration
config = ConfigParser(os.path.join(os.getcwd(), '../config.yaml'))
configs = config.get_config()

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

if __name__ == "__main__":
    logger.info(f"Starting Servo")

    sCont = servo_I2C.servo(configs['servos'])

    # Example of modding single bit
    bit = 5 #AI bit (5)
    register = sCont.MODE1
    print(f"Set Auto increment off, bit: {bit}")

    mode1_state = sCont.readReg(register) #Read the current state
    mode1_state &= ~(1<<bit) # Clear the offending bit
    #mode1_state |= (1<<bit) # Set the offending bit
    sCont.writeReg(register, mode1_state, printResp=True)
    
    servoList = [0, 1, 2,3]
    #sCont.readServoState(servoNum)

    
    servoMax = 2500
    servoMin = 500
    while(True):
        keyInput = input("Enter Servo MicroSeconds (q to quit) <enter>:")
        if keyInput == 'q':
            logger.info("Quitting on user input")
            exit()
        pulseWidth = int(keyInput)

        if pulseWidth <= servoMax and pulseWidth >= servoMin:
            print(f"Setting servo(s) to: {pulseWidth}")
            sCont.setSleep(True)
            sCont.setPulseW_us(servoList[0], pulseWidth)
            sCont.setPulseW_us(servoList[1], pulseWidth)
            sCont.setPulseW_us(servoList[2], pulseWidth)
            sCont.setPulseW_us(servoList[3], pulseWidth)
            sCont.setSleep(False)
        else:
            logger.error(f"Invalid Input. Must be < {servoMax} and > {servoMin}")


