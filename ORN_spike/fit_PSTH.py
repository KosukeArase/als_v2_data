# -*- coding: utf-8 -*-

"""
This program fits curves to the PSTH.

INPUT: PSTH data
OUTPUT: parameters

USAGE
$ python fit_PSTH.py [data_dir]
"""


import sys
import math
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit


def read_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            var = line.split()
            if len(var) > 1:
                if var[1] == "BIN":
                    bin = float(var[2])
                elif var[1] == "NUM_DATA":
                    num = int(var[2])
                    PSTH = np.empty([num])
                elif var[1] == "START_TIME":
                    start = float(var[2])
                elif var[1] == "STIMULI_DURATION":
                    duration = float(var[2])
                elif var[1] == "DOSE":
                    dose = float(var[2])
            else:
                PSTH[i-5] = float(var[0])

    return bin, num, start, duration, dose, PSTH


def spontaneous(t, f_sp):
    return f_sp


def adaptation(t, epsilon):
    return 1 - epsilon * (t - start)


def Michaelis_Menten(c, alpha, K):
    return alpha / (1 + K/c)


def rising_spike(t, tau_rise, epsilon, alpha, K):
    f_pe = adaptation(t, epsilon) * Michaelis_Menten(dose, alpha, K)
    return f_sp + f_pe * (1-np.exp(-(t-start)/tau_rise))


def falling_spike(t, f_max, tau_fall):
    # return f_sp + f_max * np.exp(-(t-(start+duration))/tau_fall)
    # return f_sp + f_max * np.exp(-(t-(start+duration))/tau_rise)
    return f_sp + f_max * (t -(start+duration) - tau_fall)**4



def optimize_parameters(time, PSTH):
    parameter_optimal ,covariance = curve_fit(spontaneous, time[:int(start/bin)], PSTH[:int(start/bin)])
    global f_sp
    f_sp = parameter_optimal[0]

    parameter_optimal ,covariance = curve_fit(rising_spike, time[int(start/bin)-1:int((start+duration)/bin)], PSTH[int(start/bin)-1:int((start+duration)/bin)])
    global tau_rise
    tau_rise, epsilon, alpha, K = parameter_optimal

    # parameter_initial = np.array([50, tau_rise]) #f_max, tau_fall
    parameter_optimal ,covariance = curve_fit(falling_spike, time[int((start+duration)/bin):], PSTH[int((start+duration)/bin):])
    f_max, tau_fall = parameter_optimal

    return f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall


if __name__ == "__main__":
    filepath = sys.argv[1]
    bin, num, start, duration, dose, PSTH = read_data(filepath)
    time = np.arange(num) * bin

    print "bin = {0}\nnum = {1}\nstart = {2}\nduration = {3}\ndose = {4}".format(bin, num, start, duration, dose)
    print "time =", time
    print "PSTH =", PSTH
    print "==================================="

    f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall = optimize_parameters(time, PSTH)
    print "f_sp = {0}\ntau_rise = {1}\nepsilon = {2}\nalpha = {3}\nK = {4}\nf_max = {5}\ntau_fall = {6}".format(f_sp, tau_rise, epsilon, alpha, K, f_max, tau_fall)

    f_before = spontaneous(time[:int(start/bin)], f_sp) * np.ones(int(start/bin))
    f_rise = rising_spike(time[int(start/bin):int((start+duration)/bin)], tau_rise, epsilon, alpha, K)
    f_fall = falling_spike(time[int((start+duration)/bin):], f_max, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))
    plt.plot(time, PSTH, "o")
    plt.plot(time, f_connected, "-")
    plt.show()


