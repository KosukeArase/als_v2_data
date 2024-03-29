# -*- coding: utf-8 -*-

"""
This program fits curves to the PSTH.

INPUT: PSTH data, dose
OUTPUT: parameters

Hint:
The INPUT is matrix (PSTH , duration and dose), so this program execute 3d fitting.
In order to realize 3d fitting, the INPUT is formed as below:
input = [[spike, spike, spike, ...], [dose, dose, dose, ...], [duration, duration, duration, ...]]

Caution!
BIN must be the same value through the input data.

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
                    start_index = int(var[2])
                elif var[1] == "STIMULI_DURATION":
                    duration = float(var[2])
                elif var[1] == "DOSE":
                    dose = float(var[2])
            else:
                time[i-5], PSTH[i-5] = map(float, var)
    return bin, num, start_index, duration, dose, PSTH, time


def save_parameters():
    output_file = "parameters_{0}ms.txt".format(int(duration*1000))
    with open(output_file, "w") as f:
        f.write("[{0}ms]\n".format(int(duration*1000)))
        f.write("f_sp = {0}\ntau_rise = {1}\nalpha = {2}\nK = {3}\ntau_fall = {4}\nmu = {5}".format(f_sp, tau_rise, alpha, K, tau_fall, mu))


def draw_fitted_curve(dose, c):
    time_rising_200[1,:] = dose * np.ones(int(duration/bin)+1)
    time_falling_200[1,:] = dose * np.ones(right - (int(duration/bin)+1))
 
    f_before = f_sp * np.ones(len(time_before_200[0]))
    f_rise = rising_spike(time_rising_200, tau_rise, alpha, K, mu)
    f_fall = falling_spike(time_falling_200, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))

    plt.plot(time, f_connected, "-", label=str(dose)+"ng_fitted_curve", color=c)


def spontaneous(t, f_sp):
    return f_sp


def Michaelis_Menten(c, alpha, K):
    return alpha / (1 + K/c)


def rising_spike(data, tau_rise, alpha, K, mu):
    f_pe = Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_pe * ((1-mu)*np.exp(-(data[0])/tau_rise) + mu)


def falling_spike(data, tau_fall):
    joint = copy.deepcopy(data)
    joint[0,:] = joint[2,:]
    fmax = rising_spike(joint, tau_rise, alpha, K, mu)
    # return f_sp + f_max * np.exp(-(data[0]-data[2]-delay)/tau_fall)
    return f_sp + fmax * np.exp(-(data[0]-data[2])/tau_fall)
    # return f_sp + f_max * np.exp(-(t-(start+duration))/tau_rise)
    # return f_sp + f_max * (data[0] - data[2]- tau_fall)**2


def get_index(x):
    return int(round(x/bin))


def optimize_parameters():
    global f_sp, tau_rise, alpha, K, mu

    parameter_optimal ,covariance = curve_fit(spontaneous, time_spontaneous, PSTH_spontaneous)
    f_sp = parameter_optimal[0]

    parameter_optimal ,covariance = curve_fit(rising_spike, time_rising, PSTH_rising)
    tau_rise, alpha, K, mu = parameter_optimal

    # parameter_initial = np.array([50, tau_rise]) #f_max, tau_fall
    parameter_optimal ,covariance = curve_fit(falling_spike, time_falling, PSTH_falling)
    tau_fall = parameter_optimal[0]

    return f_sp, tau_rise, alpha, K, tau_fall, mu


if __name__ == "__main__":
    flag_first_data = True

    input_dir = sys.argv[1]

    files = glob.glob("{0}/*.txt".format(input_dir))
    print "{0} files was imported.".format(len(files))
    for file in files:
        print file
        bin, num, start_index, duration, dose, PSTH, time_vec = read_data(file)
        stop_index = start_index + int(duration/bin) + 1
        dose_vec = np.ones(num) * dose
        duration_vec = np.ones(num) * duration
        matrix = np.vstack((time_vec, dose_vec, duration_vec))

        if flag_first_data:
            time_spontaneous = matrix[:,:start_index]
            time_rising = matrix[:,start_index:stop_index]
            time_falling = matrix[:,stop_index-1:]
            PSTH_spontaneous = PSTH[:start_index]
            PSTH_rising = PSTH[start_index:stop_index]
            PSTH_falling = PSTH[stop_index-1:]
            flag_first_data = False
        else:
            time_spontaneous = np.c_[time_spontaneous, matrix[:,:start_index]]
            time_rising = np.c_[time_rising, matrix[:,start_index:stop_index]]
            # print file
            # print matrix[:,start_index:stop_index]
            time_falling = np.c_[time_falling, matrix[:,stop_index-1:]]
            PSTH_spontaneous = np.hstack((PSTH_spontaneous, PSTH[:start_index]))
            PSTH_rising = np.hstack((PSTH_rising, PSTH[start_index:stop_index]))
            # print PSTH[start_index:stop_index]
            PSTH_falling = np.hstack((PSTH_falling, PSTH[stop_index-1:]))

        if file == "parsed_data/Park_1000ms/10000ng_10000.txt":
            time_10000 = matrix[0]
            PSTH_10000 = PSTH
        elif file == "parsed_data/Park_1000ms/3000ng_3000.txt":
            time_3000 = matrix[0]
            PSTH_3000 = PSTH
        elif file == "parsed_data/Park_1000ms/1000ng_1000.txt":
            time_1000 = matrix[0]
            PSTH_1000 = PSTH

    print "==================================="
    f_sp, tau_rise, alpha, K, tau_fall, mu = optimize_parameters()
    print "f_sp = {0}\ntau_rise = {1}\nalpha = {2}\nK = {3}\ntau_fall = {4}\nmu = {5}\n".format(f_sp, tau_rise, alpha, K, tau_fall, mu)

    save_parameters()

    left = 50
    duration = 1
    right = 150

    time_before_200 = np.vstack((bin*np.arange(-left,0), np.ndarray(left), duration * np.ones(left)))
    time_rising_200 = np.vstack((bin*np.arange(0,int(duration/bin)+1), np.ndarray(int(duration/bin)+1), duration * np.ones(int(duration/bin)+1)))
    time_falling_200 = np.vstack((bin*np.arange(int(duration/bin)+1,right), np.ndarray(right-(int(duration/bin)+1)), duration * np.ones(right-(int(duration/bin)+1))))


    """ dots """
    # time_connected = np.hstack((time_spontaneous, time_rising, time_falling))[0]
    # PSTH = np.hstack((PSTH_spontaneous, PSTH_rising, PSTH_falling))
    plt.plot(time_1000, PSTH_1000, "v", color="blue", label="1000ng_PSTH")
    plt.plot(time_3000, PSTH_3000, "o", color="red", label="3000ng_PSTH")
    plt.plot(time_10000, PSTH_10000, "x", color="green", label="10000ng_PSTH")

    """ x axis for fitted curves """
    time = bin * np.arange(-left, right)

    draw_fitted_curve(1000, "blue")
    draw_fitted_curve(5000, "red")
    draw_fitted_curve(10000, "green")
    plt.rcParams["font.size"] = 15

    plt.title("{0} ms".format(duration * 1000))
    plt.xlabel("time")
    plt.xlim(-5,10)
    plt.ylim(0,160)
    plt.ylabel("PSTH")
    plt.legend()
    plt.show()


