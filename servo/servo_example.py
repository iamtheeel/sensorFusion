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

    mode1_state = sCont.readReg(register)
    mode1_state &= ~(1<<bit) # Clear 
    sCont.writeReg(register, mode1_state, printResp=True)
    
    servoNum = 0
    sCont.readServoState(servoNum)
    waitTime = 1 #s
    numIters = 10

    # Set the pwm perioud
    pulseWidths = [650, 1500, 2500]
    for counts in list(range(1, numIters)):
      for pw in pulseWidths:
        #highBit, lowBit = sCont.servo_uSec2HB_LB(pw)
        #logger.info(f"For pulse width: {pw} uS, HB: 0x{highBit:02x}, LB:  0x{lowBit:02x}")
        #sCont.setPulseW_us(servoNum, pw)
        sCont.setPulseW_us(servoNum, pw)

        logger.info(f"wait: {waitTime}s")
        sCont.readServoState(servoNum, printVal = True)
        sleep(waitTime)


    del sCont #deconstruct
