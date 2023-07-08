
import jax.numpy as numpy
import jax
import optax
import distrax
from cpplot.cpplot import comparehist, zeroerr, divbinom, divuncorr


# get a prediction for bin probabilities
# the params are the fractions for n-1 processes.
def predictprobs(fracs , processes):
  return numpy.sum(fracs.T * processes, axis=1)


# sterling's approximation
# (not needed for multinomial...)
def logPoiss(ks , lambdas):
  return ks * numpy.log(lambdas) - lambdas - ks * numpy.log(ks) + ks


@jax.jit
def neglogLH(norms , processes , data):
  fracs = toprob(norms)

  probs = predictprobs(fracs, processes)

  # regularization to prevent negative fractions
  regularization = numpy.sum(jax.nn.softplus(-1000.0*fracs))

  return \
    regularization \
    - distrax.Multinomial(numpy.sum(data), probs=probs).log_prob(data)

  # return \
  #   - distrax.Multinomial(numpy.sum(data), probs=probs).log_prob(data)


def normalize(xs, axis=None):
  return xs / numpy.sum(xs, axis=axis)


def toprob(fracs):
  return numpy.concatenate([fracs, numpy.expand_dims(1-numpy.sum(fracs), 0)])


def fitProcFracs \
  ( procs
  , datatemp
  , nsteps=20000
  , lr=1e-3
  , gradtolerance=None
  , verbose=True
  , plotprefix=None
  , proclabels=None
  ):

  nprocs = procs.shape[1]
  startfrac = 1.0 / nprocs
  params = numpy.array( [startfrac] * (nprocs - 1) )

  optimizer = optax.adam(learning_rate=lr)
  opt_state = optimizer.init(params)

  if verbose:
    print("initial -llh:")
    print(neglogLH(params, procs, datatemp))
    print()

  for _ in range(nsteps):
    loss_value, grads = jax.value_and_grad(neglogLH)(params, procs, datatemp)

    if gradtolerance is not None:
      if numpy.all(numpy.abs(grads) < gradtolerance):
        break

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)


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

    if proclabels:
      prochists = [ zeroerr(p) for p in (fracs.T * procs).T ]

      fig = \
        comparehist \
        ( [datafrac , pred] + prochists
        , numpy.arange(datatemp.shape[0]+1)
        , [ "data" , "fit" ] + proclabels
        , "$\\tau$ width bin"
        , "binned probability density"
        , markers=["o" , ""] + [""]*len(proclabels)
        , alphas=[ 1 , 1 ] + [0.25]*len(proclabels)
        , ratio=True
        )

    else:
      fig = \
        comparehist \
        ( [datafrac , pred]
        , numpy.arange(datatemp.shape[0]+1)
        , [ "data" , "fit" ]
        , "$\\tau$ width bin"
        , "binned probability density"
        , markers=["o" , ""]
        , ratio=True
        )

    plt = fig.get_axes()[0]
    plt.legend()
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
