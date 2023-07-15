import serial
import sys
import glob
import numpy as np
from itertools import product


def setup_serial(commport, baudrate):
    """ Sets up given serial port for a given baudrate

    :param commport: target serial port
    :param baudrate: target baudrate
    :return: serial object
    """
    try:
        ser = serial.Serial(commport, baudrate, timeout=0.1)
        return ser
    except (OSError, serial.SerialException):
        raise OSError("Error setting up serial port")


def find_serial():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def acquire_data(ser, num_channel, window_size=1, EOL=None):
    """ Acquire serial port data
    Confirms validity of incoming data
    :param ser: serial object
    :param num_channel: number of distinct channels
    :param window_size: for 1D data, number of timepoints to acquire before passing
    :param EOL: end of line phrase used to separate timepoints
    :return: channel data
    """
    ser_data = np.zeros(window_size, num_channel)
    channel_data = np.array([])
    # Decode incoming data into ser_data array
    if EOL is None:
        for i in product((range(window_size)), range(num_channel)):
            try:
                ser_data[i] = ser.readline().decode().strip()
            except ValueError:
                pass
    else:
        # TODO: EOL Handler
        pass
    # If any zeros
    for i in range(window_size):
        if any(ser_data[i, :] == 0):
            pass
        else:
            np.append(channel_data, ser_data[i, :])
    if channel_data.size > 0:
        return channel_data
    else:
        return
