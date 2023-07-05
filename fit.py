
import jax.numpy as numpy
import jax
import optax
import distrax


def normalize(xs):
  return xs / numpy.sum(xs)

lJVTtemp = numpy.array([5.965314, 14.868644, 29.8614, 33.853043, 26.693424, 34.91885, 42.89062, 31.950882, 25.946457, 22.946795, 38.93199, 25.850447, 31.926224, 30.964334, 3.9892325, 6.9819517])
hJVTtemp = numpy.array([42.0337, 274.03543, 395.03888, 419.1491, 300.45645, 196.89699, 158.43587, 116.654205, 73.64295, 43.863403, 71.746735, 55.74551, 33.91508, 20.401127, 12.980115, 3.9820325])
Zmmtemp = numpy.array([26.831717, 90.72696, 167.7385, 136.61246, 112.84317, 74.0426, 44.746758, 27.41954, 17.83823, 14.77199, 16.580477, 11.576072, 3.8162308, 2.782171, 0.9710051, 0.97651887])

datatemp = numpy.array([220, 1017, 1232, 824, 494, 290, 184, 117, 127, 78, 104, 70, 89, 44, 20, 22])

# lJVTtemp = numpy.array([ 5.9734297 , 22.914124 , 51.855713 , 60.889404 , 62.71382 , 83.76105 , 74.84409 , 65.77193 , 69.9086 , 54.839725 , 90.82608 , 83.760376 , 107.71457 , 71.91319 , 21.854141 , 13.894949 ])
# hJVTtemp = numpy.array([ 49.843502 , 261.40704 , 543.7371 , 658.74194 , 617.1432 , 501.3426 , 359.51852 , 276.34143 , 193.4526 , 130.4565 , 164.3026 , 100.66779 , 94.783104 , 66.69296 , 16.960741 , 9.946595 ])
# Zmmtemp = numpy.array([ 28.858868 , 191.2353 , 334.09647 , 371.41302 , 332.84552 , 284.4817 , 190.75288 , 138.77753 , 77.75806 , 61.391243 , 66.88164 , 48.7958 , 21.946049 , 12.654993 , 5.4863586 , 0.8706384 ])

# datatemp = numpy.array([ 30 , 237 , 451 , 563 , 492 , 326 , 264 , 168 , 162 , 99 , 158 , 111 , 116 , 88 , 36 , 21 ])

procs = \
  numpy.concatenate \
  ( list(map( lambda x : numpy.expand_dims(x, axis=1) , [ lJVTtemp , hJVTtemp , Zmmtemp ] ))
  , axis=1
  )

  # logPoiss : (k : ℕ) (ν : RealVarExpr) → RealVarExpr
  # logPoiss zero ν = neg ν
  # logPoiss k ν =
  #   let k' = !! k
  #   in k' * log ν - ν - k' * log k' + k'


def toprob(fracs):
  return numpy.concatenate([fracs, numpy.expand_dims(1-numpy.sum(fracs), 0)])

# get a prediction given some normalizations and processes
# the three params are the normfactors of the processes
def predict(params , processes):
  ntot = params[0]
  fracs = toprob(params[1:])
  return ntot * numpy.sum(fracs.T * processes, axis=1)


# sterling's approximation
def logPoiss(ks , lambdas):
  return ks * numpy.log(lambdas) - lambdas - ks * numpy.log(ks) + ks


@jax.jit
def neglogLH(norms , processes , data ):
  rates = predict(norms, processes)
  return - numpy.mean(logPoiss(data , rates))


params = numpy.array([10 , 0.3 , 0.3])
# optimizer = optax.sgd(learning_rate=3e-5)
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)


print("initial -llh:")
print(neglogLH(params, procs, datatemp))
print()

for _ in range(20000):
  loss_value, grads = jax.value_and_grad(neglogLH)(params, procs, datatemp)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)


print("ntotal predicted:")
pred = predict(params, procs)
ntotal = numpy.sum(pred)
print(ntotal)
print()

print("ntotal data:")
print(numpy.sum(datatemp))
print()

print("fractions:")
print(toprob(params[1:]))
print()

print("gradients:")
print(jax.grad(neglogLH)(params, procs, datatemp))
print()

print("hessian:")
hess = jax.hessian(neglogLH)(params, procs, datatemp)
print(hess)
print()

print("covariance:")
print(numpy.linalg.inv(hess))
print()


print("final -llh:")
print(neglogLH(params, procs, datatemp))
print()

print("final prediction:")
print(predict(params, procs))
print()

print("data:")
print(datatemp)
print()

print("data/prediction:")
print(datatemp / predict(params, procs))
print()
