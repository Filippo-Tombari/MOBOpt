#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script that prints the Pareto front of a given file
"""

import numpy as np
import matplotlib.pyplot as pl

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--target", dest="target", type=str,
                    help="target function", default=None,
                    required=False)

args = parser.parse_args()


if args.target == "ZDT1":
    f1 = np.linspace(0, 1, 1000)
    f2 = 1 - np.sqrt(f1)
elif args.target == "ZDT2":
    f1 = np.linspace(0, 1, 1000)
    f2 = 1 - f1 ** 2
elif args.target == "ZDT3":
    f1 = np.linspace(0, .08300, 200)
    f1 = np.append(f1, np.linspace(.1822, .25770, 200))
    f1 = np.append(f1, np.linspace(.4093, .45380, 200))
    f1 = np.append(f1, np.linspace(.6183, .65250, 200))
    f1 = np.append(f1, np.linspace(.8233, .8518, 200))
    f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
elif args.target == "SCHAFFER":
    x = np.linspace(0, 2, 100)
    NParam = 1
    f1 = (x ** 2 )
    f2 = (x - 2) ** 2
elif args.target == "FONSECA":
    x = np.linspace(0, 1, 1000)
    f1 = 1 - np.exp(-3 * ((8 * x - 4 - 1 / np.sqrt(3)) ** 2))
    f2 = 1 - np.exp(-3 * ((8 * x - 4 + 1 / np.sqrt(3)) ** 2))

Filename1 = "SMS-EGO_" + args.target + ".dat.npz"
Filename2 = "NSGAII_" + args.target + ".dat.npz"

Data = np.load(Filename1)
F1D1, F2D1 = Data["Front"][:, 0], Data["Front"][:, 1]
I2D1 = np.argsort(F1D1)

Data = np.load(Filename2)
F1D2, F2D2 = Data["Front"][:, 0], Data["Front"][:, 1]
I2D2 = np.argsort(F1D2)


pl.plot(F1D1[I2D1], F2D1[I2D1], 'o--', label="SMS-EGO")
pl.plot(F1D2[I2D2], F2D2[I2D2], 'o--', label="NSGAII")

pl.plot(f1, f2, '-', label="TPF")
pl.legend()
pl.title(args.target)
pl.xlabel(r"$f_1$")
pl.ylabel(r"$f_2$")

pl.grid()

pl.show()
