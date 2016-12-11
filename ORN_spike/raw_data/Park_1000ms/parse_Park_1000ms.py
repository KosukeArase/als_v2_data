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
    with open(path, "r") as file:
        header = file.readline()
        lines = file.readlines()
        data = np.zeros([len(lines), 2])
        for i, line in enumerate(lines):
            data[i, 0] += float(line.split()[0]) - start
            data[i, 1] += float(line.split()[1])

    return data


def get_index(x):
    return int(math.floor(x/bin) + math.fabs(math.floor(data[0]/bin)))


def bin_round(x):
    return round(x/bin) * bin


def calc_PSTH():
    PSTH = np.zeros([num, 2])
    PSTH[:,0] = bin * np.arange(num).T + math.floor(data[0]/bin) * bin
    for spike in data:
        i = get_index(spike)
        # PSTH[i, 0] = math.floor(spike/bin) * bin
        PSTH[i, 1] += 1/bin
    return PSTH


def get_start_index():
    return int(math.fabs(round(PSTH[0,0]/bin)))


if __name__ == "__main__":
    bin = 0.1
    start = 6.0
    duration = 1.0

    # suffix = "_adjusted_spt.txt"
    target_dir = "../../parsed_data/Park_1000ms/"

    old_file = os.listdir(target_dir)
    for file in old_file:
        os.remove(target_dir + file)

    input_list = glob.glob("*.dat")
    print input_list
    for input_file in input_list:
        prefix = input_file.split(".")[0]

        if prefix[:3] == "088":
            dose = 10000
        else:
            dose = prefix.split("n")[0]
        print dose

        PSTH = read_data(input_file)

        num = len(PSTH)

        output_file = target_dir + prefix + "_" + str(dose) + ".txt"

        with open(output_file, "w") as file:
            file.write("$ BIN %s\n" % bin)
            file.write("$ NUM_DATA %d\n" % num)
            file.write("$ START_INDEX %s\n" % get_start_index())
            file.write("$ STIMULI_DURATION %s\n" % duration)
            file.write("$ DOSE %s\n" % dose)
            for t, f in PSTH:
                file.write("%s %s\n" % (t, f))
    print "Parsed data was created on {0}".format(target_dir)