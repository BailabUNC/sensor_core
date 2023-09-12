import pathlib
import sys
import multiprocessing


def setup_process_start_method():
    """ Set up multiprocessing start method based on operating system

    """
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method("spawn", force=True)
        os_flag = 'win'
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin') or sys.platform.startswith('darwin'):
        multiprocessing.set_start_method("fork", force=True)
        os_flag = 'unix'
    else:
        raise EnvironmentError('Unsupported platform')
    return os_flag

