# -*- coding: utf-8 -*-

"""
This program creates spike.

INPUT: length, start, duration, dose, parameter
OUTPUT: spiketiming [dose(ms)]

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
    alpha = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "alpha")
    K = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "K")
    tau_fall = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "tau_fall")
    mu = parameter_file.get("{0}ms".format(int(parameter_file_index*1000)), "mu")

    return map(float, [f_sp, tau_rise, alpha, K, tau_fall, mu])


def save_spiketiming(i):
    output_file = "%sspt%03d.dat" % (target_dir, i)
    spike = np.random.poisson(lam=f_connected*dt)
    spiketiming = time[0][spike != 0]
    with open(output_file, "w") as f:
        for spike in spiketiming:
            f.write("{0}\n".format(spike + start))

        f.write(str(len(spiketiming))+"\n")


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
    return f_sp + fmax * np.exp(-(data[0]-data[2])/tau_fall)


def draw_fitted_curve(dose):
    time_rising[1,:] = dose * np.ones(len(time_rising[1,:]))
    time_falling[1,:] = dose * np.ones(len(time_falling[1,:]))
    f_before = f_sp * np.ones(len(time_before[0]))
    f_rise = rising_spike(time_rising, tau_rise, alpha, K, mu)
    f_fall = falling_spike(time_falling, tau_fall)
    f_connected = np.hstack((f_before, f_rise, f_fall))
    plt.plot(time[0], f_connected, "-", label=str(dose))


if __name__ == "__main__":
    dose = int(sys.argv[1])
    parameter_file_index = 1 # 1000ms

    length = 9.1
    start = 0.1
    duration = 0.1
    interval = 0.2
    spike_times = 30

    dt = 0.000025 # 0.025ms
    # dt = 0.1

    f_sp, tau_rise, alpha, K, tau_fall, mu = load_parameters()


    """
    とりあえず時定数を10で割ってそれっぽくしてみようの会
    """
    tau_rise /= 4
    tau_fall /= 10
    """
    ここは後でなんとかしよう
    """

    print "f_sp = {0}\ntau_rise = {1}\nalpha = {2}\nK = {3}\ntau_fall = {4}\nmu = {5}".format(f_sp, tau_rise, alpha, K, tau_fall, mu)

    time = np.vstack((dt*np.arange(int(length/dt))-start, dose*np.ones(int(length/dt)), duration*np.ones(int(length/dt))))

    left = int(duration/dt)
    right = int((duration+interval)/dt)


    time_before = time[:,:int(start/dt)]

    # time_rising[1,:] = dose * np.ones(len(time_rising[1,:]))
    # time_falling[1,:] = dose * np.ones(len(time_falling[1,:]))
    f_before = f_sp * np.ones(len(time_before[0])-1)
    f_connected = f_before

    for i in xrange(spike_times):
        offset = 0#int((start+(duration+interval)*i)/dt)
        time_rising = time[:,offset:offset+left]
        time_falling = time[:,offset+left:offset+right]
        f_rise = rising_spike(time_rising, tau_rise, alpha, K, mu)
        f_fall = falling_spike(time_falling, tau_fall)
        print f_fall
        f_connected = np.hstack((f_connected, f_rise, f_fall))
        print f_connected
        print len(f_connected)

    print len(time[0])
    plt.plot(time[0], f_connected, "-", label=str(dose))


    """ spike 周りの拡大用
    time_before = time_before[:,-1000:]
    time_falling = time_falling[:,:1000]
    time = time[:,int(start/dt)-1000:int((start+duration)/dt)+1000]
    """

    # draw_fitted_curve(10000)
    # draw_fitted_curve(5000)
    # draw_fitted_curve(1000)

    plt.title("{0} ms".format(dose))
    plt.xlabel("time")
    plt.ylabel("PSTH")
    plt.legend()

    plt.show()

    # f_before = f_sp * np.ones(len(time_before[0]))

    # f_rise = rising_spike(time_rising, tau_rise, alpha, K, mu)
    # f_fall = falling_spike(time_falling, tau_fall)
    # f_connected = np.hstack((f_before, f_rise, f_fall))
    # print f_before[-5:]
    # print f_rise[:5]
    # print f_rise[-5:]
    # print f_fall[:5]
    # print f_fall[-5:]

    # print np.average(f_before)
    # print np.average(f_rise)
    # print np.average(f_fall)
    # print np.average(f_connected)
    num_spike_file = 1000
    target_dir = "{0}dose_30stims/ORN/".format(dose)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for i in xrange(num_spike_file):
        # print spiketiming
        save_spiketiming(i)
