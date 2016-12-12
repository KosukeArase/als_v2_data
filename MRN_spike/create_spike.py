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


def save_spiketiming():
    output_file = "spiketiming/mechano_{0}ms_{1}Hz.txt".format(int(duration*1000), freq)
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
    freq = 20
    duration = 1

    length = 20
    start = 5
    dt = 0.000025


    n = int(length/dt)
    time = dt * np.arange(n)


    spike = np.random.poisson(lam=dt*freq*np.ones(n))
    spike[int(start/dt):int((start+duration)/dt)] = 0

    spiketiming = time[spike != 0]

    print spiketiming
    save_spiketiming()


    # time = np.vstack((dt*np.arange(int(length/dt))-start, dose*np.ones(int(length/dt)), duration*np.ones(int(length/dt))))

    # left = int(start/dt)
    # right = int((start+duration)/dt)


    # time_before = time[:,:left]
    # time_rising = time[:,left:right]
    # time_falling = time[:,right:]


    # f_before = f_sp * np.ones(len(time_before[0]))

    # f_rise = rising_spike(time_rising, tau_rise, epsilon, alpha, K, delay, beta)
    # f_fall = falling_spike(time_falling, tau_fall)
    # f_connected = np.hstack((f_before, f_rise, f_fall))
    # # print f_before[-5:]
    # # print f_rise[:5]
    # # print f_rise[-5:]
    # # print f_fall[:5]
    # # print f_fall[-5:]

    # # print np.average(f_before)
    # # print np.average(f_rise)
    # # print np.average(f_fall)
    # # print np.average(f_connected)

    # spike = np.random.poisson(lam=f_connected*dt)

    # spiketiming = time[0][spike != 0]

    # # print spiketiming
