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


def load_parameters():
    parameter_file_path = "parameters_" + str(int(parameter_file_index*1000)) + "ms.txt"
    parameter_file = ConfigParser.SafeConfigParser()
    parameter_file.read(parameter_file_path)
    print parameter_file_path


    alpha = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "alpha")
    K = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "K")
    t_delay = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "t_delay")
    tau_rise = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_rise")
    tau_plateau = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_plateau")
    tau_fall = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_fall")
    mu = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "mu")
    f_sp = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "f_sp")

    return map(float, [alpha, K, t_delay, tau_rise, tau_plateau, tau_fall, mu, f_sp])


def save_spiketiming(n):
    spike = np.random.poisson(lam=fitted_curve*dt)
    spiketiming = mat[0][spike != 0]
    output_file = "%sspt%03d.dat" % (target_dir, i)
    with open(output_file, "w") as f:
        for j, spike in enumerate(spiketiming):
            f.write("{0}\n".format(spike - t_left))

        f.write(str(len(spiketiming))+"\n")


def draw_fitted_curve(dose, color):
    time_v = np.arange(-500,1000)/100.
    dose_v = np.ones(1500) * dose
    dur_v = np.ones(1500) * duration
    mat = np.vstack((time_v, dose_v, dur_v))
    mat[0,:] += t_left

    fitted_curve = optimize_parameters(mat, t_delay, tau_rise, tau_plateau, tau_fall, mu)

    plt.plot(time_v, fitted_curve, "-", label=str(dose)+"ng_fitted_curve", color=color)


def raster(event_times_list, height, color='k'):
    """
    Creates a raster plot

    Parameters
    ----------
    event_times_list : iterable
                       a list of event time iterables
    color : string
            color of vlines
    Returns
    -------
    ax : an axis containing the raster plot
    """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, 0+height, 40+height, color=color)
    # plt.ylim(.5, 2)
    return ax


def spontaneous(t, f_sp):
    return f_sp * np.ones(len(t[0]))


def Michaelis_Menten(dose, alpha, K): # [dose] => freq
    return alpha / (1 + K/dose)


def rise(data, tau_rise, alpha, K, t_start):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    # return f_sp + f_peak * (2/(1+np.exp(-tau_rise*(data[0]-t_start)))-1) # sigmoid
    return f_sp + f_peak * np.exp(tau_rise*(data[0]-t_peak)) # exp


def plateau(data, tau_plateau, mu):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_peak * ((1-mu)*np.exp(-data[0]*tau_plateau) + mu)


def fall(data, tau_fall, mu, t_stop):
    f_peak = Michaelis_Menten(data[1], alpha, K)
    return f_sp + mu * f_peak * np.exp(-(data[0]-t_stop)*tau_fall)


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
    dose = 3000
    dt = 0.000025 # 0.025ms
    duration = 1.
    bin = 0.1

    t_peak = 0.
    t_start = t_peak - bin # -0.1

    t_left = -0.5
    t_right = 9.5
    # t_right = 4.5

    parameter_file_index = 1 # 1000ms

    alpha, K, t_delay, tau_rise, tau_plateau, tau_fall, mu, f_sp = load_parameters()
    f_sp *= 2.

    print "alpha = {0}\nK = {1}\nt_delay = {2}\ntau_rise = {3}\ntau_plateau = {4}\ntau_fall = {5}\nmu = {6}\nf_sp = {7}".format(alpha, K, t_delay, tau_rise, tau_plateau, tau_fall, mu, f_sp)

    time_v = np.arange(t_left/dt,t_right/dt) * dt
    dose_v = np.ones(len(time_v)) * dose
    dur_v = np.ones(len(time_v)) * duration
    mat = np.vstack((time_v, dose_v, dur_v))

    fitted_curve = optimize_parameters(mat, t_delay, tau_rise, tau_plateau, tau_fall, mu)
    # fitted_curve[:int(len(fitted_curve)/100)] = 100
    # print fitted_curve[:10]

    """ save spiketiming"""
    num_spike_file = 1000
    target_dir = "spiketiming_plateau/{0}ng_1stim/".format(dose)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for i in xrange(num_spike_file):
        # print spiketiming
        print i
        save_spiketiming(i)

    # draw_fitted_curve(10000)
    # draw_fitted_curve(5000)
    draw_fitted_curve(dose, "blue")



    """ draw raster """
    filename = "./spiketiming_plateau/3000ng_1stim/spt000.dat"

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = map(float, lines)
        del lines[-1]

        ax = raster(lines, 0)
    filename = "../MRN_spike/spiketiming_plateau/30Hz_1stim/spt000.dat"

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = map(float, lines)
        del lines[-1]

        ax = raster(lines, 70)
    """ end raster """


    plt.title("{0} ms, {1} ng".format(int(duration*1000), dose))
    plt.xlabel("time")
    plt.ylabel("PSTH")
    plt.legend()
    plt.rcParams["font.size"] = 15
    plt.xlim(-0.5,4.5)

    plt.show()

