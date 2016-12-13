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
import numpy as np


def read_data(path):
    with open(path, "r") as file:
        header = file.readline().split(",")
        start = float(header[0].split()[-1])
        stop = float(header[1].split()[-1])

        lines = file.readlines()

        data = np.zeros([len(lines)])

        for i, line in enumerate(lines):
            data[i] += float(line.split()[0])

        print data

    return start, stop, data


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

if __name__ == "__main__":
    bin = 0.1
    duration = 0.2

    suffix = "_adjusted_spt.txt"
    target_dir = "../../parsed_data/Park_200ms/"
    
    old_file = os.listdir(target_dir)
    for file in old_file:
        os.remove(target_dir + file)

    with open("datalist.csv", "r") as datalist:
        input_list = datalist.readlines()
        for input_file in input_list:
            prefix = input_file.split(".")[0]
            dose = int(input_file.split(",")[1])
            file = prefix + suffix
            print file
            start, stop, data = read_data(file)

            num = int(math.fabs(math.floor(data[0]/bin))+math.ceil(data[-1]/bin)) + 1
            PSTH = calc_PSTH()
            # print PSTH

            output_file = target_dir + prefix + "_" + str(dose) + ".txt"

            with open(output_file, "w") as file:
                file.write("$ BIN %s\n" % bin)
                file.write("$ NUM_DATA %d\n" % num)
                file.write("$ START_INDEX %s\n" % get_index(0))
                file.write("$ STIMULI_DURATION %s\n" % duration)
                file.write("$ DOSE %s\n" % dose)
                for t, f in PSTH:
                    file.write("%s %s\n" % (t, f))
        print "Parsed data was created on {0}".format(target_dir)