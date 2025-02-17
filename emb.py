#!/usr/bin/python3
import time
import serial
from config import *
from utils import *

serial_port = serial.Serial(
    port="/dev/ttyTHS0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

while True:
    _,frame = cap.read()
    AI_Result = img_, direction_return, angle_str = pipeline_function(frame, INTEREST_BOX, paint = True, lane_paint = True, interest_box = True)
    push_str = f"#{direction_return}:{angle_str}!\r\n"
    serial_port.write(push_str.encode())
    time.sleep(0.1)