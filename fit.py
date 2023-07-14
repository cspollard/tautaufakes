
import jax.numpy as numpy
import jax.scipy.optimize as optimize
import jax
import distrax
from cpplot.cpplot import comparehist, comparehistratio, zeroerr, divbinom

from matplotlib import figure


# get a prediction for bin probabilities
# the params are the fractions for n-1 processes.
def predictprobs(fracs , processes):
  return numpy.sum(fracs.T * processes, axis=1)


# sterling's approximation
# (not needed for multinomial...)
def logPoiss(ks , lambdas):
  return ks * numpy.log(lambdas) - lambdas - ks * numpy.log(ks) + ks


def toprob(fracs):
  return jax.nn.softmax(fracs)


@jax.jit
def neglogLH(norms , processes , data):
  fracs = toprob(norms)

  probs = predictprobs(fracs, processes)

  return \
    - distrax.Multinomial(numpy.sum(data), probs=probs).log_prob(data)


def normalize(xs, axis=None):
  return xs / numpy.sum(xs, axis=axis)



def fitProcFracs \
  ( procs
  , datatemp
  , verbose=True
  , plotprefix=None
  , proclabels=None
  ):

  nprocs = procs.shape[1]
  params = numpy.zeros(nprocs)

  if verbose:
    print("initial -llh:")
    print(neglogLH(params, procs, datatemp))
    print()


  result = optimize.minimize(neglogLH , params , method="BFGS" , args=(procs, datatemp))
  params = result.x

  hess = jax.hessian(neglogLH)(params, procs, datatemp)
  cov = numpy.linalg.inv(hess)

  fracs = toprob(params)
  predfrac = normalize(predictprobs(fracs, procs))
  datafrac = normalize(datatemp)

  if verbose:
    print("ntotal data:")
    print(numpy.sum(datatemp))
    print()

    print("fractions:")
    print(fracs)
    print()

    print("gradients:")
    print(jax.grad(neglogLH)(params, procs, datatemp))
    print()

    # print("hessian:")
    # print(hess)
    # print()

    print("covariance:")
    print(cov)
    print()

    print("final -llh:")
    print(neglogLH(params, procs, datatemp))
    print()

    print("predicted fractions:")
    print(predfrac)
    print()

    print("data fractions:")
    print(datafrac)
    print()

    print("data/prediction:")
    print(datafrac / predfrac)
    print()


  if plotprefix is not None:
    datafrac = divbinom(datatemp, datatemp.at[:].set(numpy.sum(datatemp)))
    pred =  zeroerr(predfrac)

    fig = figure.Figure((8, 8))
    fig.add_subplot(3, 1, (1, 2))
    fig.add_subplot(3, 1, 3)

    prochists = [ zeroerr(p) for p in (fracs.T * procs).T ]

    proclabs = \
      [ proclab + "(%0.2f%%)" % (fracs[i] * 100) for i , proclab in enumerate(proclabels) ]

    comparehist \
      ( fig.axes[0]
      , [datafrac , pred] + prochists
      , numpy.arange(datatemp.shape[0]+1)
      , [ "data" , "fit" ] + proclabs
      , "$\\tau$ width bin"
      , "binned probability density"
      , markers=["o" , ""] + [""]*len(proclabs)
      , alphas=[ 1 , 1 ] + [0.25]*len(proclabs)
      )


    comparehistratio \
      ( fig.axes[1]
      , [datafrac , pred] + prochists
      , numpy.arange(datatemp.shape[0]+1)
      , [ "data" , "fit" ] + proclabs
      , "$\\tau$ width bin"
      , "binned probability density"
      , markers=["o" , ""] + [""]*len(proclabs)
      , alphas=[ 1 , 1 ] + [0.25]*len(proclabs)
      )

    plt = fig.get_axes()[0]
    plt.legend()
    plt.set_xticks([])
    plt.set_xticklabels([])
    plt.set_xlabel("")
    plt = fig.get_axes()[1]
    plt.set_ylim((0.5, 2))
    fig.savefig(plotprefix + "datafitcomp.pdf")


  return params , cov


if __name__ == "__main__":

  lJVTtemp = normalize(numpy.array([5.965314, 14.868644, 29.8614, 33.853043, 26.693424, 34.91885, 42.89062, 31.950882, 25.946457, 22.946795, 38.93199, 25.850447, 31.926224, 30.964334, 3.9892325, 6.9819517]))
  hJVTtemp = normalize(numpy.array([42.0337, 274.03543, 395.03888, 419.1491, 300.45645, 196.89699, 158.43587, 116.654205, 73.64295, 43.863403, 71.746735, 55.74551, 33.91508, 20.401127, 12.980115, 3.9820325]))
  Zmmtemp = normalize(numpy.array([26.831717, 90.72696, 167.7385, 136.61246, 112.84317, 74.0426, 44.746758, 27.41954, 17.83823, 14.77199, 16.580477, 11.576072, 3.8162308, 2.782171, 0.9710051, 0.97651887]))

  datatemp = numpy.array([220, 1017, 1232, 824, 494, 290, 184, 117, 127, 78, 104, 70, 89, 44, 20, 22])

  procs = \
    numpy.concatenate \
    ( list(map( lambda x : numpy.expand_dims(x, axis=1) , [ lJVTtemp , hJVTtemp , Zmmtemp ] ))
    , axis=1
    )

  fitProcFracs(procs, datatemp)
