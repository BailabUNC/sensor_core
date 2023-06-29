import serial
import numpy as np


def setup_serial(commport, baudrate):
    try:
        ser = serial.Serial(commport, baudrate, timeout=0.1)
        return ser
    except:
        print("Error in setting up serial port")


def acquire_data(ser, num_channel):
    channel_data = np.zeros(num_channel)
    for i in range(num_channel):
        try:
            channel_data[i] = ser.readline().decode().strip()
        except:
            pass
    if sum(channel_data) == 0:
        return
    else:
        return channel_data
