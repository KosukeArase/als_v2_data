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
    f_sp = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "f_sp")
    tau_rise = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_rise")
    epsilon = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "epsilon")
    alpha = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "alpha")
    K = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "K")
    delay = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "delay")
    tau_fall = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_fall")
    beta = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "beta")

    return map(float, [f_sp, tau_rise, epsilon, alpha, K, delay, tau_fall, beta])


def save_spiketiming():
    output_file = "spiketiming/{0}ng_{1}ms.txt".format(dose, int(duration*1000))
    f_sp, tau_rise, epsilon, alpha, K, delay, tau_fall
    with open(output_file, "w") as f:
        for spike in spiketiming:
            f.write("{0}\n".format(spike + start))

        f.write(str(len(spiketiming))+"\n")


def spontaneous(t, f_sp):
    return f_sp


def adaptation(t, epsilon, beta):
    return 1 - epsilon * (t - beta)


def Michaelis_Menten(c, alpha, K):
    return alpha / (1 + K/c)


def rising_spike(data, tau_rise, epsilon, alpha, K, delay, beta):
    f_pe = adaptation(data[0], epsilon, beta) * Michaelis_Menten(data[1], alpha, K)
    return f_sp + f_pe * (1-np.exp(-(data[0]-delay)/tau_rise))


# def falling_spike(data, f_max, tau_fall):
def falling_spike(data, tau_fall):
    fmax = rising_spike(data[:,0], tau_rise, epsilon, alpha, K, delay, beta)
    # return f_sp + f_max * np.exp(-(data[0]-data[2]-delay)/tau_fall)
    return f_sp + fmax * np.exp(-(data[0]-data[2]-delay)/tau_fall)
    # return f_sp + f_max * np.exp(-(t-(start+duration))/tau_rise)
    # return f_sp + f_max * (data[0] - data[2]- tau_fall)**2

def draw_fitted_curve(dose):
    time_rising[1,:] = dose * np.ones(len(time_rising[1,:]))
    time_falling[1,:] = dose * np.ones(len(time_falling[1,:]))
    f_before = f_sp * np.ones(len(time_before[0]))
    f_rise = rising_spike(time_rising, tau_rise, epsilon, alpha, K, delay, beta)
    f_fall = falling_spike(time_falling, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))
    plt.plot(time[0], f_connected, "-", label=str(dose))


if __name__ == "__main__":
    dose = 10000
    duration = 0.2

    length = 20
    start = 5
    parameter_file_index = 1 # 1000ms
    dt = 0.000025
    # dt = 0.1

    f_sp, tau_rise, epsilon, alpha, K, delay, tau_fall, beta = load_parameters()

    print "f_sp = {0}\ntau_rise = {1}\nepsilon = {2}\nalpha = {3}\nK = {4}\ndelay = {5}\ntau_fall = {6}\nbeta = {7}".format(f_sp, tau_rise, epsilon, alpha, K, delay, tau_fall, beta)


    time = np.vstack((dt*np.arange(int(length/dt))-start, dose*np.ones(int(length/dt)), duration*np.ones(int(length/dt))))

    left = int(start/dt)
    right = int((start+duration)/dt)


    time_before = time[:,:left]
    time_rising = time[:,left:right]
    time_falling = time[:,right:]

    """ spike 周りの拡大用
    time_before = time_before[:,-1000:]
    time_falling = time_falling[:,:1000]
    time = time[:,int(start/dt)-1000:int((start+duration)/dt)+1000]
    """

    draw_fitted_curve(10000)
    draw_fitted_curve(5000)
    draw_fitted_curve(1000)

    plt.title("{0} ms".format(int(duration*1000)))
    plt.xlabel("time")
    plt.ylabel("PSTH")
    plt.legend()

    plt.show()

    f_before = f_sp * np.ones(len(time_before[0]))

    f_rise = rising_spike(time_rising, tau_rise, epsilon, alpha, K, delay, beta)
    f_fall = falling_spike(time_falling, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))
    # print f_before[-5:]
    # print f_rise[:5]
    # print f_rise[-5:]
    # print f_fall[:5]
    # print f_fall[-5:]

    # print np.average(f_before)
    # print np.average(f_rise)
    # print np.average(f_fall)
    # print np.average(f_connected)

    spike = np.random.poisson(lam=f_connected*dt)

    spiketiming = time[0][spike != 0]

    # print spiketiming
    save_spiketiming()