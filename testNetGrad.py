from data_util import prepareData, splitData
from layers import *
from gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from nnet import FullyConnectedNet
import matplotlib.pyplot as plt
from solver import Solver
import time
from relativeError import *
import numpy as np

print('\n--------- test multilayer nnet loss and grad --------- ')

np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 1
X = np.random.randn(N, D)
#y = np.random.randint(C, size=(N,))
y = np.random.randn(N, C)

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)
  print('X shape: ', X.shape, 'y shape: ', y.shape)
  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_maxError(grad_num, grads[name])))
