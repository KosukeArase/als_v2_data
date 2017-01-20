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
import numpy as np


def save_spiketiming(i):
    target_dir = "general_odor_{0}Hz/".format(frequency)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    output_file = "%sspt%03d.dat" % (target_dir, i)

    with open(output_file, "w") as f:
        for spike in spiketiming:
            f.write("{0}\n".format(spike))

        f.write(str(len(spiketiming))+"\n")

if __name__ == "__main__":
    dt = 0.000025 # 0.025ms

    length = 10
    frequency = 10

    n = int(length/dt)
    time = dt * np.arange(n)
    print n

    for i in xrange(1000):
        spike = np.random.poisson(lam=dt*frequency*np.ones(n))
        spiketiming = time[spike != 0]
        save_spiketiming(i)
        print i
