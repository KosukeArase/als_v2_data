# -*- coding: utf-8 -*-

"""
This program parse raw data suitable for fit_PSTH.py

INPUT: Files listed on datalist.csv
OUTPUT: parsed data

USAGE
$ python parse.py [output_dir]
"""

import os
import sys
import math
import glob
import numpy as np


def read_data(path):
    prefix = input_file.split(".")[0]

    if prefix[:3] == "088":
        dose = 10000
    else:
        dose = prefix.split("n")[0]
        # print dose
    print path
    with open(path, "r") as file:
        var = file.readline()
        num_spike = int(file.readline().split(":")[1])
        start_num = file.readline().split()[5]
        var = file.readline()
        var = file.readline()
        var = file.readline()

        lines = file.readlines()[:num_spike]
        data = np.zeros(num_spike)

        flag = 0
        threshold = 0.05
        for i, line in enumerate(lines):
            if len(line.split()) != 5:
                break
            data[i] = float(line.split()[3])
            if line.split()[1] == start_num:
                stimuli_start = data[i]
                start_index = i
                flag = 1
            if flag == 1 and stimuli_start < data[i-2] < stimuli_start + 0.4 and i >= 2 and data[i] - data[i-2] < threshold: # is the beginning of spike burst?
                start = data[i-2]
                start_index = i
                flag = 2

    if flag == 0:
        print "'#Start time of stimulation' was not found in .dat file."
    elif flag == 1:
        start = stimuli_start
        print "Spikes dense enough was not found."

    print "start = {0}".format(start)

    stop = start + duration
    data = np.delete(data, np.where(data==0.)[0], 0)
    data = data - start

    return dose, start, stop, start_index, data


def get_index(x):
    return int(math.floor(x/bin) + math.fabs(math.floor(data[0]/bin)))


def bin_round(x):
    return round(x/bin) * bin


def calc_PSTH():
    PSTH = np.zeros([num, 2])
    PSTH[:,0] = bin * np.arange(num).T + math.floor(data[0]/bin) * bin
    for spike in data:
        i = get_index(spike)
        PSTH[i, 1] += 1/bin
    return PSTH


if __name__ == "__main__":
    bin = 0.1
    start = 6.0
    duration = 1.0

    target_dir = "../../parsed_data/Park_1000ms/"

    old_file = os.listdir(target_dir)
    for file in old_file:
        os.remove(target_dir + file)

    input_list = glob.glob("*.dat")
    for input_file in input_list:
        dose, start, stop, start_index, data = read_data(input_file)

        num = int(math.fabs(math.floor(data[0]/bin)) + math.ceil(data[-1]/bin)) + 1
        PSTH = calc_PSTH()

        prefix = input_file.split(".")[0]
        output_file = target_dir + prefix + "_" + str(dose) + ".txt"

        with open(output_file, "w") as file:
            file.write("$ BIN %s\n" % bin)
            file.write("$ NUM_DATA %d\n" % num)
            file.write("$ START_INDEX %s\n" % get_index(0))
            file.write("$ STIMULI_DURATION %s\n" % duration)
            file.write("$ DOSE %s\n" % dose)
            for t, f in PSTH:
                file.write("%s %s\n" % (t, f))
        print "===================="
    print "Parsed data was created on {0}".format(target_dir)