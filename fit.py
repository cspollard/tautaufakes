
import jax.numpy as numpy
import jax
import optax
import distrax


# get a prediction given some normalizations and processes
# the first parameter is the total normalization;
# the next two params are the lJVT and hJVT fractions, respectively.
def predict(params , processes):
  fracs = toprob(params)
  return numpy.sum(fracs.T * processes, axis=1)


# sterling's approximation
# (not needed for multinomial...)
def logPoiss(ks , lambdas):
  return ks * numpy.log(lambdas) - lambdas - ks * numpy.log(ks) + ks


@jax.jit
def neglogLH(norms , processes , data):
  probs = predict(norms, processes)
  return - distrax.Multinomial(numpy.sum(data), probs=probs).log_prob(data)


def normalize(xs, axis=None):
  return xs / numpy.sum(xs, axis=axis)


def toprob(fracs):
  return numpy.concatenate([fracs, numpy.expand_dims(1-numpy.sum(fracs), 0)])


def fitProcFracs(procs, datatemp, nsteps=20000, lr=1e-3):

  nprocs = procs.shape[1]
  startfrac = 1.0 / nprocs
  params = numpy.array( [startfrac] * (nprocs - 1) )

  optimizer = optax.adam(learning_rate=lr)
  opt_state = optimizer.init(params)

  print("initial -llh:")
  print(neglogLH(params, procs, datatemp))
  print()

  for _ in range(nsteps):
    loss_value, grads = jax.value_and_grad(neglogLH)(params, procs, datatemp)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

  print("ntotal data:")
  print(numpy.sum(datatemp))
  print()

  print("fractions:")
  print(toprob(params))
  print()

  print("gradients:")
  print(jax.grad(neglogLH)(params, procs, datatemp))
  print()

  print("hessian:")
  hess = jax.hessian(neglogLH)(params, procs, datatemp)
  print(hess)
  print()

  print("covariance:")
  cov = numpy.linalg.inv(hess)
  print(cov)
  print()

  print("final -llh:")
  print(neglogLH(params, procs, datatemp))
  print()

  print("predicted fractions:")
  predfrac = normalize(predict(params, procs))
  print(predfrac)
  print()

  print("data fractions:")
  datafrac = normalize(datatemp)
  print(datafrac)
  print()

  print("data/prediction:")
  print(datafrac / predfrac)
  print()

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
