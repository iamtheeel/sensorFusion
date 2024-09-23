# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Fall 2024
#
###
#
# Control a servo: https://en.wikipedia.org/wiki/Servo_control
# Using a PCA9685 16CH 12Bit PWM Led Driver
# 
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

from time import sleep

## Configure I2C
from periphery import I2C
# Connected to I2C2 which is /dev/i2c-1
# There is sombody at 1, nobody at 0 or 2
i2c = I2C("/dev/i2c-1")
device = 0x40

## Configure Timing
center_us = 1500
full_CW_us = 2000
full_CCW_us = 1000

#Register List
MODE1 = [0x00] 
MODE2 = [0x01]
PRESC = [0xFE] # Timer prescailer


def readReg(regToRead):
    msgs = [I2C.Message(regToRead), I2C.Message([0x00], read = True)]
    i2c.transfer(device, msgs)
    #print(f"Read Reg: 0x{msgs[0].data[0]:02x}, 0b{msgs[1].data[0]:08b}, 0x{msgs[1].data[0]:02x}")
    sleep(0.1)
    return msgs[1].data[0]

def writeReg(regToWrite, newVal, readAfter=False):
    msgs = [I2C.Message(regToWrite + [newVal], read = False)]
    i2c.transfer(device, msgs)
    sleep(0.1)

    #This does not work
    if readAfter:
        newVal = readReg(regToRead)
        return newVal

    
#register = [0x00]
register = MODE1

# Read
read_1 = readReg(register)
print(f"Initial read: 0x{register[0]:02x}, 0b{read_1:08b}, 0x{read_1:02x}")


# Write
#msgs = [I2C.Message(PRESC + [0x1E])] # Array of elements --> [Register, Message]

#newVal = msgs[1].data[0] |  (1<<7) # Reset
#newVal = msgs[1].data[0] & ~(1<<0) # Turn off all call
#newVal = read_1 |  (1<<0) # Turn on ALLCALL

newVal = read_1 |  (1<<4) # Turn on SLEEP
writeReg(register, newVal )
read_2 = readReg(register)
print(f"after write : 0x{register[0]:02x}, 0b{read_2:08b}, 0x{read_2:02x}")

'''
#newVal = msgs[1].data[0] | (1<<4) # Turn off sleep mode
print(f"TrnOn Sleep :       0b{newVal:08b}")
msgs = [I2C.Message(register + [newVal], read = False)]
i2c.transfer(device, msgs)
sleep(0.1)
'''



'''
# To set to internal clock:
newVal = msgs[1].data[0] |  (1<<4) # Turn on sleep mode
print(f"val to send :       0b{newVal:08b}")
msgs = [I2C.Message(register + [newVal], read = False)]
i2c.transfer(device, msgs)
sleep(0.1)
newVal &= ~(1<<6) # Set to EXTCLK to internal 
print(f"val to send :       0b{newVal:08b}")
msgs = [I2C.Message(register + [newVal], read = False)]
i2c.transfer(device, msgs)
sleep(0.1)

# Read
msgs = [I2C.Message(register), I2C.Message([0x00], read = True)]
i2c.transfer(device, msgs)
print(f" check value: 0x{msgs[0].data[0]:02x}, 0b{msgs[1].data[0]:08b}, 0x{msgs[1].data[0]:02x}")
'''


# LED Registers
LED0_ON_L  = [0x06] 
LED0_ON_H  = [0x07] 
LED0_OFF_L = [0x08] 
LED0_OFF_H = [0x09] 

# Read the LED
LED0_ON_L_State = readReg(LED0_ON_L)
LED0_ON_H_State = readReg(LED0_ON_H)
LED0_OFF_L_State = readReg(LED0_OFF_L)
LED0_OFF_H_State = readReg(LED0_OFF_H)
print(f"LED0_ON_L Initial : 0x{LED0_ON_L[0]:02x}, 0b{LED0_ON_L_State:08b}, 0x{LED0_ON_L_State:02x}")
print(f"LED0_ON_H Initial : 0x{LED0_ON_H[0]:02x}, 0b{LED0_ON_H_State:08b}, 0x{LED0_ON_H_State:02x}")
print(f"LED0_OFF_L Initial : 0x{LED0_OFF_L[0]:02x}, 0b{LED0_OFF_L_State:08b}, 0x{LED0_OFF_L_State:02x}")
print(f"LED0_OFF_H Initial : 0x{LED0_OFF_H[0]:02x}, 0b{LED0_OFF_H_State:08b}, 0x{LED0_OFF_H_State:02x}")


read_1 = readReg(MODE1)
newVal = read_1 |  (1<<4) # Turn on SLEEP
writeReg(MODE1, newVal )
read_2 = readReg(MODE1)
print(f"Sleep ON : 0x{register[0]:02x}, 0b{read_2:08b}, 0x{read_2:02x}")

### Make the changes
# Note: all registers need to be written
writeReg(LED0_ON_L, 0x99)
writeReg(LED0_ON_H, 0x1)
writeReg(LED0_OFF_L, 0xCC)
writeReg(LED0_OFF_H, 0x4)


### Turn sleep back off
read_1 = readReg(MODE1)
newVal = read_1 & ~(1<<4) # Turn off SLEEP
writeReg(MODE1, newVal )
read_2 = readReg(MODE1)
print(f"Sleep OFF : 0x{register[0]:02x}, 0b{read_2:08b}, 0x{read_2:02x}")

# Set new values
#Recheck
LED0_ON_L_State = readReg(LED0_ON_L)
LED0_ON_H_State = readReg(LED0_ON_H)
LED0_OFF_L_State = readReg(LED0_OFF_L)
LED0_OFF_H_State = readReg(LED0_OFF_H)
print(f"LED0_ON_L New : 0x{LED0_ON_L[0]:02x}, 0b{LED0_ON_L_State:08b}, 0x{LED0_ON_L_State:02x}")
print(f"LED0_ON_H New : 0x{LED0_ON_H[0]:02x}, 0b{LED0_ON_H_State:08b}, 0x{LED0_ON_H_State:02x}")
print(f"LED0_OFF_L New : 0x{LED0_OFF_L[0]:02x}, 0b{LED0_OFF_L_State:08b}, 0x{LED0_OFF_L_State:02x}")
print(f"LED0_OFF_H New : 0x{LED0_OFF_H[0]:02x}, 0b{LED0_OFF_H_State:08b}, 0x{LED0_OFF_H_State:02x}")



i2c.close()
