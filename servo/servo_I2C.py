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
#
# Notes:
#
#
###

from periphery import I2C
from math import floor

## Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from time import sleep

class servo:
    #Register List
    MODE1 = 0x00 
    MODE2 = 0x01
    PRESC = 0xFE # Timer prescailer

    # Servo Registers (LED in doc)
    SERVO_BASE = {"ON_L": 0x06, "ON_H": 0x07, "OFF_L": 0x08, "OFF_H": 0x09} # we expect a list hence the [0xYY]

    def __init__(self, configs):
        self.i2c_device = configs['i2c']['device']
        self.servoRate = configs['servos']['pwm_Hz']

        # Init the I2C device
        self.i2c_port = I2C(configs['i2c']['port'])

        self.config = configs #Stash the rest of the configs for later use

        #restert, extclk, Auto Increemnt, SLEEP, res, res, res, ALLCALL
        # Note you can not turn off extclock by setting value = 0, must reset board
        self.readReg(self.MODE1, printResp=True)
        self.readReg(self.MODE2, printResp=True)

        self.setPSC(self.servoRate, displayVal=True)

        #logger.info(f"Done with configs, setting sleep off")
        self.setSleep(False)

    def __del__(self):
        # Set Sleep
        logger.info(f"Shutting down")

        if self.config['servos']['leavRunning']:
            logger.info(f"Leaving the servos running")
        else:
            self.setSleep(True)
        self.i2c_port.close()

    def readReg(self, regToRead, printResp=False):
        msgs = [I2C.Message([regToRead]), I2C.Message([0x00], read = True)]
        self.i2c_port.transfer(self.i2c_device, msgs)

        if printResp:
            logger.info(f"readRge 0x{regToRead:02x}: 0b{msgs[1].data[0]:08b}, 0x{msgs[1].data[0]:02x}")
        else:
            sleep(0.1)

        return msgs[1].data[0]

    def writeReg(self, regToWrite, newVal, returnState=False, printResp=False):
        msgs = [I2C.Message([regToWrite] + [newVal], read = False)]
        self.i2c_port.transfer(self.i2c_device, msgs)
        sleep(0.1)

        if returnState or printResp:
            readVal = self.readReg(regToWrite)

        if printResp:
            logger.info(f"after write: 0x{regToWrite:02x}: 0b{readVal:08b}, 0x{readVal:02x}")

        if returnState:
            return readVal

    def getServoAddresses(self, servo_num):
        ON_L_ADDR = self.SERVO_BASE['ON_L']+4*servo_num
        ON_H_ADDR = self.SERVO_BASE['ON_H']+4*servo_num
        OFF_L_ADDR = self.SERVO_BASE['OFF_L']+4*servo_num
        OFF_H_ADDR = self.SERVO_BASE['OFF_H']+4*servo_num

        return ON_L_ADDR, ON_H_ADDR, OFF_L_ADDR, OFF_H_ADDR

    def readServoState(self, servo_num, printVal = False):
        # Get the addresses
        ON_L_ADDR, ON_H_ADDR, OFF_L_ADDR, OFF_H_ADDR = self.getServoAddresses(servo_num)

        ON_L_State = self.readReg(ON_L_ADDR)
        ON_H_State = self.readReg(ON_H_ADDR)
        OFF_L_State = self.readReg(OFF_L_ADDR)
        OFF_H_State = self.readReg(OFF_H_ADDR)

        if printVal:
            print(f"Servo{servo_num}_ON_L : 0x{ON_L_ADDR:02x}, 0b{ON_L_State:08b}, 0x{ON_L_State:02x}")
            print(f"Servo{servo_num}_ON_H : 0x{ON_H_ADDR:02x}, 0b{ON_H_State:08b}, 0x{ON_H_State:02x}")
            print(f"Servo{servo_num}_OFF_L: 0x{OFF_L_ADDR:02x}, 0b{OFF_L_State:08b}, 0x{OFF_L_State:02x}")
            print(f"Servo{servo_num}_OFF_H: 0x{OFF_H_ADDR:02x}, 0b{OFF_H_State:08b}, 0x{OFF_H_State:02x}")

        return ON_L_State, ON_H_State, OFF_L_State, OFF_H_State

    def setSleep(self, sleepState):
        mode1_state = self.readReg(self.MODE1)
        if sleepState == False:
            mode1_state &= ~(1<<4) # Turn off SLEEP
        else:
            mode1_state |=  (1<<4) # Turn on SLEEP

        logger.debug(f"val to write: 0b{mode1_state:08b}, 0x{mode1_state:02x}")
        self.writeReg(self.MODE1, mode1_state, printResp=False)

    # To restart, clear sleep bit, then set restart bit
    #def restart:
        #newVal = read_1 &  ~(1<<4) # Turn off SLEEP
        #newVal = msgs[1].data[0] |  (1<<7) # Reset

    def setPSC(self, update_rate, displayVal=False):
        #prescale value PSC =  round(osc_clock/(4096*update_rate)) - 1
        # 4095 = 2^12 for 12 bit clock
        PSC = round(self.config['i2c']['clock_MHz'] * 1e6 /(pow(2,12) * update_rate) -1)
        #logger.info(f"Setting new PSC: 0x{PSC:02x}")

        self.setSleep(True) # sleep has to be on to program the prescailer
        ## Set the PSC
        self.writeReg(self.PRESC, PSC, printResp=displayVal)
        self.setSleep(False)


    def servo_uSec2HB_LB(self, uSec):
        timeStep_us = (1/self.servoRate)/pow(2,12)*1e6
        #logger.info(f"timeStep: {timeStep_us}us")
        counts = int(round(uSec/timeStep_us, 0))
        over8bit = counts/pow(2,8)
        #logger.info(f"uSec: {uSec}, counts: {counts}, over8bit: {over8bit}")

        highBit = floor(over8bit)
        lowBit = counts - highBit*pow(2,8)

        return highBit, lowBit

    def setPulseW_us(self, servo_num, uSec):
        hb, lb = self.servo_uSec2HB_LB(uSec)

        #logger.info(f"For servo: {servo_num}, Time: {uSec}uS, HB: 0x{hb:02x}, LB:  0x{lb:02x}")
        self.setPulseW_HB_LB(servo_num, hb, lb)

    def setPulseW_HB_LB(self, servo_num, HB, LB):
        #logger.info(f"setPulseW_HB_LB: {servo_num}")
        ON_L_ADDR, ON_H_ADDR, OFF_L_ADDR, OFF_H_ADDR = self.getServoAddresses(servo_num)\

        #We must write all 4 registers for each servo every time
        #self.setSleep(True)
        self.writeReg(ON_L_ADDR, 0x00) # We want 0 delay
        self.writeReg(ON_H_ADDR, 0x00)
        self.writeReg(OFF_L_ADDR, LB)
        self.writeReg(OFF_H_ADDR, HB)
        #self.setSleep(False)


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

