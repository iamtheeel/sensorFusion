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
import gpiod
CONSUMER = "servo"
logger.info(f"Configuring pin: {CONSUMER}")
chip_4 = gpiod.Chip("4", gpiod.Chip.OPEN_BY_NUMBER)
GPIO_P36 = chip_4.get_line(13)  # Chip, Line = 4, 13
GPIO_P36.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

## Configure Timing
import time
pulseLen_ms = 1000
#pulseLen_ms = 20
center_us = 1500
full_CW_us = 2000
full_CCW_us = 1000

dutyTime_s = 0.2

# convert to usefull units
pulseLen_s = pulseLen_ms/1000

startTime_s = time.time()
while True:
    
    # Set the pins duty cycle
    #logger.info(f"Time 1: {time.time()}")
    duty_elapsedTime_s = time.time() - startTime_s
    sleepTime_s = dutyTime_s - duty_elapsedTime_s
    # sleep to the duty cycle
    if sleepTime_s <0: sleepTime_s = 0
    time.sleep(sleepTime_s)
    dutyLen_meas_ms = (time.time() - startTime_s) * 1000
    logger.error(f"Duty Time: {dutyLen_meas_ms:0.3f} ms")
    GPIO_P36.set_value(False)

    # Go to the end of the pulse len
    pulse_elapsedTime_s = time.time() - startTime_s
    sleepTime_s = pulseLen_s - pulse_elapsedTime_s
    if sleepTime_s <0: sleepTime_s = 0
    time.sleep(sleepTime_s)
    #logger.debug(f"Time 3: {time.time()}")
    GPIO_P36.set_value(True)
    pulseLen_meas_ms = (time.time() - startTime_s) * 1000

    #Reset the timer immediatly after GPIO(True)
    startTime_s = time.time()

    # Debuging
    #logger.error(f"Start time: {startTime_s} ")
    #logger.error(f"Pulse Len: {pulseLen_ms:0.3f} miliSec")
    #logger.error(f"Elapsed Time, Duty makeup: {(duty_elapsedTime_s*1000:0.3f} miliSec")
    logger.error(f"Elapsed Time, Pulse: {pulse_elapsedTime_s*1000:0.3f} miliSec")
    logger.error(f"SleepTime: {sleepTime_s*1000:0.3f} miliSec")
    logger.error(f"pulseLen: {pulseLen_meas_ms:0.3f} miliSec")
    logger.error("-----------------------------------------")


# On exit let go
GPIO_P36.release()
