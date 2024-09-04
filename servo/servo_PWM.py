# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Fall 2024
#
###
#
# Control a servo: https://en.wikipedia.org/wiki/Servo_control
#
# The GPIO Pins are organised by "Chip" and "Line" https://coral.ai/docs/dev-board/gpio/#header-pinout
###

## Configuration
import sys, os
sys.path.insert(0, '..')  # ConfigParser is in the project root
from ConfigParser import ConfigParser
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


## Configure GPIO
from periphery import PWM

servo_a = PWM(1,0) # Chip, Channel
servo_a.frequency = 50
servo_a.enable()

## Configure Timing
import time
#pulseLen_ms = 20
center_us = 1500
full_CW_us = 2000
full_CCW_us = 1000

stepSize = 100
stepDir = 1

pulseTime_us = center_us
dutyCycl = pulseTime_us 
servo_a.duty_cycle = 0.1

while True:
    # convert to usefull units
    if pulseTime_us > full_CW_us: 
        stepDir = -1
        pulseTime_us = full_CW_us
    elif pulseTime_us < full_CCW_us: 
        stepDir = 1
        pulseTime_us = full_CCW_us

    dutyCycl = servo_a.frequency * (pulseTime_us /1000/1000)
    #logger.error(f"Pulse Time: {pulseTime_us}us, dutyCycle: {dutyCycl}%")
    servo_a.duty_cycle = dutyCycl
    pulseTime_us += stepSize*stepDir
    time.sleep(1.5)
