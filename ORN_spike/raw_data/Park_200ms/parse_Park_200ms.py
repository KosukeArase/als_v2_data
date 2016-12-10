# -*- coding: utf-8 -*-

"""
This program parse raw data suitable for fit_PSTH.py

INPUT: raw PTSH data
OUTPUT: parsed data

USAGE
$ python parse.py [input_dir] [output_dir]
"""


import sys
import math


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


