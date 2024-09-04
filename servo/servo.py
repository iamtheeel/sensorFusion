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

import gpiod


# Configure pins
CONSUMER = "servo"
chip_4 = gpiod.Chip("4", gpiod.Chip.OPEN_BY_NUMBER)
GPIO_P36 = chip_4.get_line(13)  # Chip, Line = 4, 13
GPIO_P36.request(consumer=CONSUMER, type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

# Working with the pin
GPIO_P36.set_value(0)

