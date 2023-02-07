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

target = targets.target(args.target.lower(),NParam)
f1 = target.f1
f2 = target.f2
PB = target.PB
func = target.func
iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
GenDist = []
Delta = []
for NIter in iterations:
    Optimize = mo.MOBayesianOpt(target=func,
                                NObj=2,
                                pbounds=PB,
                                Picture=False,
                                MetricsPS=False,
                                TPF=np.asarray([f1, f2]).T,
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
