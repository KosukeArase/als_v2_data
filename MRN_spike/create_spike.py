# -*- coding: utf-8 -*-

"""
This program creates spike.

INPUT: length, start, duration, dose, parameter
OUTPUT: spiketiming

USAGE
$ python create_spike.py
"""


import os
import sys
import math
import copy
import glob
import numpy as np
import ConfigParser
import matplotlib.pylab as plt
from scipy.optimize import curve_fit


def save_spiketiming(i):
    if spike_times == 1:
        target_dir = "spiketiming/{0}Hz_1stim/".format(frequency)
    else:
        target_dir = "spiketiming/{0}Hz_{1}stims/".format(frequency, spike_times)
    # target_dir = "unko_m/".format(frequency)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    output_file = "%sspt%03d.dat" % (target_dir, i)

    with open(output_file, "w") as f:
        for spike in spiketiming:
            f.write("{0}\n".format(spike))

        f.write(str(len(spiketiming))+"\n")


def draw_fitted_curve(dose):
    time_rising[1,:] = dose * np.ones(len(time_rising[1,:]))
    time_falling[1,:] = dose * np.ones(len(time_falling[1,:]))
    f_before = f_sp * np.ones(len(time_before[0]))
    f_rise = rising_spike(time_rising, tau_rise, epsilon, alpha, K, delay, beta)
    f_fall = falling_spike(time_falling, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))
    plt.plot(time[0], f_connected, "-", label=str(dose))


if __name__ == "__main__":
    dt = 0.000025 # 0.025ms

    start = 0
    duration = 1
    interval = 9
    spike_times = 1
    length = start + (duration + interval) * spike_times

    frequency = 60

    n = int(length/dt)
    time = dt * np.arange(n)
    print n

    for i in xrange(1000):
        spike = np.random.poisson(lam=dt*frequency*np.ones(n))
        for j in xrange(spike_times):
            offset = int((start+(duration+interval)*j)/dt)
            spike[offset:offset+int(duration/dt)] = 0
        spiketiming = time[spike != 0]

        save_spiketiming(i)
        print i
