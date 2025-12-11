import serial
import sys
import glob
import numpy as np
from typing import *
from itertools import product


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


class SerialManager:
    def __init__(self, commport: str, baudrate: int, frame_shape: Tuple[int, ...],
                 EOL: str = None, virtual_ser_port: bool = False):
        """ Initialize SerialManager class - manages functions related to instantiating and using serial port

        :param commport: target serial port
        :param baudrate: target baudrate
        :param frame_shape: shape of output data (number of distinct channels, number of frames)
        :param EOL: optional; end of line phrase used to separate timepoints
        :param virtual_ser_port: boolean, if True will not initialize serial port, instead will rely on user-defined
        custom function to generate simulated data
        """
        self.commport = commport
        self.baudrate = baudrate
        self.frame_shape = frame_shape
        self.EOL = EOL
        self.ser = None
        self.virtual_ser_port = virtual_ser_port

    def setup_serial(self):
        """ Sets up given serial port for a given baudrate

        :return: serial object
        """
        if self.virtual_ser_port:
            pass
        else:
            try:
                self.ser = serial.Serial(self.commport, self.baudrate, timeout=0.1)
                return self.ser
            except (OSError, serial.SerialException):
                raise OSError("Error setting up serial port")

    def _acquire_data(self, frame_shape, data_mode):
        """Default (fallback) reader; replace with real protocol."""
        if data_mode == "line":
            N = int(frame_shape[0])  # num_points
            C = int(frame_shape[2])  # channels
            return np.zeros((N, C), dtype=np.float32)
        else:
            H, W, Cimg = (frame_shape[0], frame_shape[1], frame_shape[2] if len(frame_shape) == 3 else 1)
            return np.zeros((H, W, Cimg), dtype=np.float32)

    def acquire_data(self, func=None, data_mode: str='line'):
        """
        Acquire serial data
        :param func: custom acquisition function
        :param data_mode: Line or Image data acquisition
        Returns:
          LINE mode:  (S, C) float32
          IMAGE mode: (H, W, C) uint8/float32
        """
        if func is None:
            channel_data = self._acquire_data(frame_shape=self.frame_shape,
                                              data_mode=data_mode)  # your legacy path for line
        else:
            try:
                # supply expected frame shape to custom func
                expected = getattr(self, "frame_shape", (self.frame_shape[0], self.frame_shape[2]))
                channel_data = func(
                    ser=self.ser,
                    frame_shape=expected,
                )
            except Exception:
                raise ValueError(
                    f'custom function {func} must accept: ser, frame_shape'
                )

        if channel_data is None:
            return None

        arr = np.asarray(channel_data)

        # Line mode
        if data_mode == "line":
            # normalize to (S, C)
            if arr.ndim != 2:
                raise ValueError(f"LINE mode expects 2D (S, C), got {arr.shape}")
            S, C = arr.shape
            if C != self.num_channel and S == self.num_channel:
                arr = arr.T
            if arr.shape != (self.frame_shape[1], self.num_channel):
                raise ValueError(f"LINE data shape {arr.shape} != ({self.frame_shape[1]}, {self.num_channel})")
            return arr.astype(np.float32, copy=False)
        else: # Image mode
            if arr.ndim == 2:
                arr = arr[:, :, None]
            Hexp, Wexp, Cexp = self.frame_shape
            if arr.shape != (Hexp, Wexp, Cexp):
                raise ValueError(f"IMAGE data shape {arr.shape} != expected {(Hexp, Wexp, Cexp)}")
            return arr