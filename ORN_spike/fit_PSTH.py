# -*- coding: utf-8 -*-

"""
This program fits curves to the PSTH.

INPUT: PSTH data, dose
OUTPUT: parameters

Hint:
The INPUT is 2d data (PSTH and dose), so this program execute 3d fitting.
In order to realize 3d fitting, the INPUT is formed as below:
input = [[spike, spike, spike, ...], [dose, dose, dose, ...]]

Caution!
BIN must be the same value.

USAGE
$ python fit_PSTH.py [data_dir]
"""


import os
import sys
import math
import copy
import glob
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit


def read_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            var = line.split()
            if var[0] == "$":
                if var[1] == "BIN":
                    bin = float(var[2])
                elif var[1] == "NUM_DATA":
                    num = int(var[2])
                    PSTH = np.empty([num])
                    time = np.empty([num])
                elif var[1] == "START_INDEX":
                    start = int(var[2])
                elif var[1] == "STIMULI_DURATION":
                    duration = float(var[2])
                elif var[1] == "DOSE":
                    dose = float(var[2])
            else:
                time[i-5], PSTH[i-5] = map(float, var)
    return bin, num, start, duration, dose, PSTH, time


def spontaneous(t, f_sp):
    return f_sp


def adaptation(t, epsilon):
    return 1 - epsilon * (t - start)


def Michaelis_Menten(c, alpha, K):
    return alpha / (1 + K/c)


def rising_spike(data, tau_rise, epsilon, alpha, K):
    f_pe = adaptation(data[0], epsilon) * Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_pe * (1-np.exp(-data[0]/tau_rise))


def falling_spike(data, f_max, tau_fall):
    # return f_sp + f_max * np.exp(-(data[0] - data[2])/tau_fall)
    # return f_sp + f_max * np.exp(-(t-(start+duration))/tau_rise)
    return f_sp + f_max * (data[0] - data[2]- tau_fall)**2


def get_index(x):
    return int(round(x/bin))


def optimize_parameters():
    parameter_optimal ,covariance = curve_fit(spontaneous, time_spontaneous, PSTH_spontaneous)
    global f_sp
    f_sp = parameter_optimal[0]

    parameter_optimal ,covariance = curve_fit(rising_spike, time_rising, PSTH_rising)
    global tau_rise
    tau_rise, epsilon, alpha, K = parameter_optimal

    # parameter_initial = np.array([50, tau_rise]) #f_max, tau_fall
    parameter_optimal ,covariance = curve_fit(falling_spike, time_falling, PSTH_falling)
    f_max, tau_fall = parameter_optimal

    return f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall


if __name__ == "__main__":
    # time_spontaneous = np.ndarray([2, 0])
    # time_rising = np.ndarray([2, 0])
    # time_falling = np.ndarray([2, 0])

    # PSTH_spontaneous = np.ndarray([1, 0])
    # PSTH_rising = np.ndarray([1, 0])
    # PSTH_falling = np.ndarray([1, 0])

    flag_first_data = True

    input_dir = sys.argv[1]
    files = glob.glob("{0}/*.txt".format(input_dir))
    print "{0} files was imported.".format(len(files))
    for file in files:
        bin, num, start, duration, dose, PSTH, time_vec = read_data(file)
        stop = start + int(duration/bin) + 1
        # time_vec = np.arange(num) * bin
        dose_vec = np.ones(num) * dose
        # start_vec = np.zeros(num)
        duration_vec = np.ones(num) * duration
        matrix = np.vstack((time_vec, dose_vec, duration_vec))

        if flag_first_data:
            time_spontaneous = matrix[:,:start]
            time_rising = matrix[:,start:stop]
            time_falling = matrix[:,stop-1:]
            PSTH_spontaneous = PSTH[:start]
            PSTH_rising = PSTH[start:stop]
            PSTH_falling = PSTH[stop-1:]
            flag_first_data = False
        else:
            time_spontaneous = np.c_[time_spontaneous, matrix[:,:start]]
            time_rising = np.c_[time_rising, matrix[:,start:stop]]
            # print matrix[:,start:stop]
            time_falling = np.c_[time_falling, matrix[:,stop-1:]]
            PSTH_spontaneous = np.hstack((PSTH_spontaneous, PSTH[:start]))
            PSTH_rising = np.hstack((PSTH_rising, PSTH[start:stop]))
            # print PSTH[start:stop]
            PSTH_falling = np.hstack((PSTH_falling, PSTH[stop-1:]))

    # print np.average(time_spontaneous[1])
    # print time_spontaneous[0,:100]
    print time_rising[0,:100]
    # print PSTH_rising[:100]
    # print time_falling[0,:100]

    # print "bin = {0}\nnum = {1}\nstart = {2}\nduration = {3}\ndose = {4}".format(bin, num, start, duration, dose)
    # print "time =", time
    # print "PSTH =", PSTH
    # print len(PSTH)
    print "==================================="

    f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall = optimize_parameters()
    print "f_sp = {0}\ntau_rise = {1}\nepsilon = {2}\nalpha = {3}\nK = {4}\nf_max = {5}\ntau_fall = {6}".format(f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall)

    f_before = f_sp * np.ones(len(time_spontaneous[0]))
    # spontaneous(time[:get_index(start)], f_sp) * np.ones(get_index(start))
    f_rise = rising_spike(time_rising, tau_rise, epsilon, alpha, K)
    f_fall = falling_spike(time_falling, f_max, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))

    time = np.hstack((time_spontaneous, time_rising, time_falling))[0]
    # print time
    # print f_connected
    PSTH = np.hstack((PSTH_spontaneous, PSTH_rising, PSTH_falling))
    # print len(f_connected)
    # print len(time)

    plt.plot(time, PSTH, "o")
    plt.plot(time, f_connected, "-")
    plt.show()


