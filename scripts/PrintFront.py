#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script that prints the Pareto front of a given file
"""

import numpy as np
import matplotlib.pyplot as pl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest="Filename", type=str,
                    help="Filename with front")

parser.add_argument("--tpf", dest="TPF", type=str,
                    help="true Pareto front", default=None,
                    required=False)

args = parser.parse_args()

if args.TPF is not None:
    F1, F2 = np.loadtxt(args.TPF, unpack=True)
    ISorted = np.argsort(F1)
    f1 = F1[ISorted]
    f2 = F2[ISorted]
else:
    f1 = np.linspace(0, 1, 1000)
    # computation of the multiplying constant of f2
    g = 1
    n = len(f1)
    for i in range(3, n):
        g = g + (9 * f1[i]) / (n - 2)

    f2 = g * (1 - np.sqrt(f1))

if args.Filename[-3:] == "npz":
    Data = np.load(args.Filename)
    F1D2, F2D2 = Data["Front"][:, 0], Data["Front"][:, 1]
    I2 = np.argsort(F1D2)

else:
    raise TypeError("Extension should be npz")

pl.plot(F1D2[I2], F2D2[I2], 'o--', label="Bayes")
#if args.TPF is not None:
pl.plot(f1, f2, 'o', label="TPF")
pl.legend()

pl.xlabel(r"$f_1$")
pl.ylabel(r"$f_2$")

pl.grid()

pl.show()
