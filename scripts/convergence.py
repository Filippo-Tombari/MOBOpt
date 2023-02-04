# Convergence analysis of the SMS-EGO solution
import numpy as np
import matplotlib.pyplot as pl
import targets
import mobopt as mo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="ND", type=int, metavar="ND",
                    help="Number of Dimensions",
                    default=30,
                    required=False)
parser.add_argument("-ni", dest="NInit", type=int, metavar="NInit",
                    help="Number of initialization points",
                    required=False, default=10)
parser.add_argument("-np", dest="npts", type=int, metavar="npts",
                    help="Number of random points to sample at each iteration",
                    required=False, default=1000)
parser.add_argument("--target", dest="target", type=str,
                    default="ZDT1",
                    help="Target function name")
parser.set_defaults(Reduce=False)

args = parser.parse_args()

NParam = args.ND
N_init = args.NInit
n_pts = args.npts

if args.target == "ZDT1":
    target = targets.zdt1
    f1 = np.linspace(0, 1, 1000)
    f2 = 1 - np.sqrt(f1)
    PB = np.asarray([[0, 1]] * NParam)
elif args.target == "ZDT2":
    target = targets.zdt2
    f1 = np.linspace(0, 1, 1000)
    f2 = 1 - f1 ** 2
    PB = np.asarray([[0, 1]] * NParam)
elif args.target == "ZDT3":
    target = targets.zdt3
    f1 = np.linspace(0, .08300, 200)
    f1 = np.append(f1, np.linspace(.1822, .25770, 200))
    f1 = np.append(f1, np.linspace(.4093, .45380, 200))
    f1 = np.append(f1, np.linspace(.6183, .65250, 200))
    f1 = np.append(f1, np.linspace(.8233, .8518, 200))
    f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
    PB = np.asarray([[0, 1]] * NParam)
elif args.target == "SCHAFFER":
    target = targets.schaffer_mo
    x = np.linspace(-1000, 1000, 10000)
    NParam = 1
    f1 = x**2
    f2 = (x - 2)**2
    PB = np.asarray([[-1000, 1000]] * NParam)
elif args.target == "FONSECA":
    target = targets.fonseca
    NParam = 3
    x = np.linspace(-4, 4, 1000)
    f1 = 1 - np.exp( -3*( (x-1/np.sqrt(3) )**2 ))
    f2 = 1 - np.exp( -3*( (x+1/np.sqrt(3) )**2 ))
    PB = np.asarray([[-4, 4]] * NParam)
else:
    raise TypeError("Target function not available")

iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
GenDist = []
Delta = []
for NIter in iterations:

    Optimize = mo.MOBayesianOpt(target=target,
                                NObj=2,
                                pbounds=PB,
                                MetricsPS = False,
                                max_or_min='min',
                                RandomSeed=10)

    Optimize.initialize(init_points=N_init)

    front, pop = Optimize.maximize_smsego(n_iter=NIter, n_pts=n_pts)

    GenDist.append(mo.metrics.GD(-front, np.asarray([f1, f2]).T))
    Delta.append(mo.metrics.Spread2D(-front, np.asarray([f1, f2]).T))


fig, ax = pl.subplots(1, 1)
ax.plot(iterations, GenDist, '-', label="GenDist")
ax.plot(iterations,Delta,'-', label="Delta")
ax.grid()
ax.set_xlabel("Number of iterations")
ax.set_ylabel("Metric value")
ax.legend()
fig.savefig("convergence.png", dpi=300)
