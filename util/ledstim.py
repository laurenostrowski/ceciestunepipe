import numpy as np
import pandas as pd
import warnings
import logging
from itertools import groupby
from operator import itemgetter
from matplotlib import pyplot as plt
from typing import Union

logger = logging.getLogger("ceciestunepipe.util.ledstim")


## calibration 06/05/2023 for bird 
#pwm duty [1-255]
pwm_list_20230605 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 
            20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 90,
           100, 120, 150, 170, 180, 200, 225]

#readout of thorlabs power meter
pwr_list_20230605 = [0.168, 0.267, 0.351, 0.429, 0.5, .055, 0.6, 0.64, 0.685, 0.724,
           0.754, 0.781, 0.811, 0.839, 0.864,
            0.969, 1.004, 1.05, 1.12, 1.19, 1.247, 1.301, 1.351, 1.398, 6.69, 7.07, 8.9,
           9.59, 10.91, 12.85, 14.1, 37, 41.2, 45.7]

cal_array_20230605 = np.vstack([pwm_list_20230605, pwr_list_20230605])


def pwm2pwr(x:np.array, cal_array: np.array=cal_array_20230605) -> np.array:
    # gets an array of pwm duty
    return np.interp(x, *cal_array)