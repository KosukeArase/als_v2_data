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
$ python fit_PSTH.py
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
    return f_sp * np.ones(len(t[0]))


def Michaelis_Menten(dose, alpha, K): # [dose] => freq
    return alpha / (1 + K/dose)


def rise(data, tau_rise, alpha, K, t_start):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_peak * (2/(1+np.exp(-tau_rise*(data[0]-t_start)))-1) # sigmoid


def plateau(data, tau_plateau, mu):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_peak * ((1-mu)*np.exp(-data[0]*tau_plateau) + mu)


def fall(data, tau_fall, mu, t_stop):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    return f_sp + mu * f_peak * np.exp(-(data[0]-t_stop)*tau_fall)


def get_index(x):
    return int(round(x/bin))


def optimize_parameters(matrix, t_delay, tau_rise, tau_plateau, tau_fall, mu):
    # global t_peak, t_start, t_stop
    t_stop = t_peak + duration + t_delay # 1.0 + delay

    matrix_spontaneous = matrix[:, matrix[0]<t_start] # t < t_start
    matrix_rise = matrix[:,np.where((matrix[0]>=t_start)&(matrix[0]<t_peak))] # t_start <= t < t_peak
    matrix_plateau = matrix[:,np.where((matrix[0]>=t_peak)&(matrix[0]<t_stop))] # t_peak <= t < t_stop
    matrix_fall = matrix[:, t_stop<=matrix[0]] # t_stop < t


    output_spontaneous = spontaneous(matrix_spontaneous, f_sp)
    output_rise = rise(matrix_rise, tau_rise, alpha, K, t_start)
    output_plateau = plateau(matrix_plateau, tau_plateau, mu)
    output_fall = fall(matrix_fall, tau_fall, mu, t_stop)

    output = np.ndarray([len(matrix[0])])

    output[matrix[0]<t_start] = output_spontaneous
    output[np.where((matrix[0]>=t_start)&(matrix[0]<t_peak))] = output_rise
    output[np.where((matrix[0]>=t_peak)&(matrix[0]<t_stop))] = output_plateau
    output[t_stop<=matrix[0]] = output_fall

    return output


if __name__ == "__main__":
    bin = 0.1
    t_peak = 0.
    t_start = t_peak - bin # -0.1

    flag_first_data = True

    input_dir = "parsed_data/Park_1000ms/"

    files = glob.glob("{0}/*.txt".format(input_dir))
    print "{0} files was imported.".format(len(files))

    max_PSTH = np.ndarray([len(files), 2])

    for i, file in enumerate(files):
        print file
        bin, num, start_index, duration, dose, _PSTH, time_vec = read_data(file)
        stop_index = start_index + int(duration/bin) + 1
        dose_vec = np.ones(num) * dose
        duration_vec = np.ones(num) * duration
        _matrix = np.vstack((time_vec, dose_vec, duration_vec))

        max_PSTH[i] = [dose, _PSTH[start_index]]

        if flag_first_data:
            matrix = copy.deepcopy(_matrix)
            PSTH = copy.deepcopy(_PSTH)
            flag_first_data = False
        else:
            matrix = np.c_[matrix, _matrix]
            PSTH = np.hstack((PSTH, _PSTH))

        """ for PSTH plot"""
        if file == "parsed_data/Park_1000ms/10000ng_10000.txt":
            time_10000 = matrix[0]
            PSTH_10000 = PSTH
        elif file == "parsed_data/Park_1000ms/3000ng_3000.txt":
            time_3000 = matrix[0]
            PSTH_3000 = PSTH
        elif file == "parsed_data/Park_1000ms/1000ng_1000.txt":
            time_1000 = matrix[0]
            PSTH_1000 = PSTH

    # print max_PSTH
    print "==================================="


    parameter_optimal, covariance = curve_fit(Michaelis_Menten, max_PSTH[:,0], max_PSTH[:,1])
    alpha, K = parameter_optimal

    sp_indexes = np.where(matrix[0]<t_start)

    parameter_optimal, covariance = curve_fit(spontaneous, matrix[:,sp_indexes], PSTH[sp_indexes])
    f_sp = parameter_optimal[0]

    parameter_optimal, covariance = curve_fit(optimize_parameters, matrix, PSTH, p0=[0.3, 50., 10., 10., 0.4], bounds=([0,0,0,0,0], [1,100,100,100,1]))

    t_delay, tau_rise, tau_plateau, tau_fall, mu = parameter_optimal


    print "alpha = {0}\nK = {1}\nt_delay = {2}\ntau_rise = {3}\ntau_plateau = {4}\ntau_fall = {5}\nmu = {6}\nf_sp = {7}".format(alpha, K, t_delay, tau_rise, tau_plateau, tau_fall, mu, f_sp)


    time_v = np.arange(-10,40)/10.
    dose_v = np.ones(50) * 3000
    dur_v = np.ones(50) * 1.0
    mat = np.vstack((time_v, dose_v, dur_v))

    fitted_curve = optimize_parameters(mat, t_delay, tau_rise, tau_plateau, tau_fall, mu)

    plt.plot(time_v, fitted_curve, "-")
    plt.show()















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


