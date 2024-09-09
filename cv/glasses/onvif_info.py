#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Fall 2024
#
###
#
# Get ONVIF information on camera
#
###


from onvif2 import ONVIFCamera
from zeep.transports import Transport

'''
Ports   80: Bad response
        443: Refused
        554: No Response
        580: Refused
        8080: Refused
'''
mycam = ONVIFCamera('192.168.1.254', 443, 'Sunglasses DV-1a60', '12345678', '/home/josh//Documents/MIC/sensorFusion/cv/glasses/')
