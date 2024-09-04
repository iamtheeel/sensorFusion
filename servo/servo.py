# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Fall 2024
#
###
#
# Controll a servo
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

import gpiod


# Configure pins
CONSUMER = "servo"
logger.info(f"Configuring pin: {CONSUMER}")
chip_4 = gpiod.Chip("4", gpiod.Chip.OPEN_BY_NUMBER)
GPIO_P36 = chip_4.get_line(13)  # Chip, Line = 4, 13
GPIO_P36.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

# Working with the pin
GPIO_P36.set_value(1)

# On exit let go
GPIO_P36.release()
