from data_util import prepareData, splitData
from layers import *
from gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from nnet import FullyConnectedNet
import matplotlib.pyplot as plt
from solver import Solver
import time
from relativeError import rel_meanError
import numpy as np

print('\n--------- square_loss & crossEntropy_loss test --------- ')

np.random.seed(231)
num_classes, num_inputs = 1, 50
x = np.random.randn(num_inputs, num_classes)
y = np.random.randn(num_inputs, num_classes)

print('\nx: ', x)
print('\ny: ', y)
dx_num = eval_numerical_gradient(lambda x: square_loss(x, y)[0], x, verbose=False)
loss, dx = square_loss(x, y)
#print('\ndx_num: ', dx_num)
#print('\ndx: ', dx)
print('\nnp.c_[dx_num, dx]: ', np.c_[dx_num, dx])
# Test square_loss function. 
print('Testing square_loss:')
print('loss: ', loss)
print('dx error: ', rel_meanError(dx_num, dx))

#------------------------------------------------------------------------------
#%%
print('\n--------- affine_sigmoid_affine_backward test --------- ')
np.random.seed(231)
x = np.random.randn(10, 10)
w1 = np.random.randn(10, 5)
b1 = np.random.randn(5)
w2 = np.random.randn(5, 7)
b2 = np.random.randn(7)
dout = np.random.randn(10, 7)
dout1 = np.random.randn(10, 5)


out, cache = affine_sigmoid_affine_forward(x, w1, b1, w2, b2)
dx, dw1, db1, dw2, db2 = affine_sigmoid_affine_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda x: affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], x, dout)
dw1_num = eval_numerical_gradient_array(lambda w1: affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], w1, dout)
db1_num = eval_numerical_gradient_array(lambda b1: affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], b1, dout)
dw2_num = eval_numerical_gradient_array(lambda w2: affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], w2, dout)
db2_num = eval_numerical_gradient_array(lambda b2: affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], b2, dout)    
#out, cache = affine_sigmoid_forward(x, w1, b1)
#dx, dw, db = affine_sigmoid_backward(dout1, cache)
#dx_num = eval_numerical_gradient_array(lambda x: 
#                    affine_sigmoid_forward(x, w1, b1)[0], x, dout1)
    

print('\ndx_num: ', dx_num)
print('\ndx: ', dx)

print('Testing sigmoid_backward function:')
print('dx error: ', rel_meanError(dx_num, dx))
print('dw1 error: ', rel_meanError(dw1_num, dw1))
print('db1 error: ', rel_meanError(db1_num, db1))
print('dw2 error: ', rel_meanError(dw2_num, dw2))
print('db2 error: ', rel_meanError(db2_num, db2))
#------------------------------------------------------------------------------
#%%
print('\n--------- affine_sigmoid_affine_squareloss_backward test --------- ')
np.random.seed(231)
x = np.random.randn(10, 10)
w1 = np.random.randn(10, 5)
b1 = np.random.randn(5)
w2 = np.random.randn(5, 1)
b2 = np.random.randn(1)
dout = np.random.randn(10, 1)
y = np.random.randn(10, 1)
dout1 = np.random.randn(10, 5)


loss, out, cache= affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)
dx, dw1, db1, dw2, db2 = affine_sigmoid_affine_squareloss_backward(cache)

dx_num  = eval_numerical_gradient(lambda x:  affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)[0], x)
dw1_num = eval_numerical_gradient(lambda w1: affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)[0], w1)
db1_num = eval_numerical_gradient(lambda b1: affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)[0], b1)
dw2_num = eval_numerical_gradient(lambda w2: affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)[0], w2)
db2_num = eval_numerical_gradient(lambda b2: affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y)[0], b2)
   

print('\ndx_num: ', dx_num)
print('\ndx: ', dx)

print('Testing sigmoid_backward function:')
print('dx error: ', rel_meanError(dx_num, dx))
print('dw1 error: ', rel_meanError(dw1_num, dw1))
print('db1 error: ', rel_meanError(db1_num, db1))
print('dw2 error: ', rel_meanError(dw2_num, dw2))
print('db2 error: ', rel_meanError(db2_num, db2))
#%%
print('\n--------- affine_backward test --------- ')
np.random.seed(231)
x = np.random.randn(10, 10)
w1 = np.random.randn(10, 5)
b1 = np.random.randn(5)
w2 = np.random.randn(5, 7)
b2 = np.random.randn(7)
dout = np.random.randn(10, 7)
dout1 = np.random.randn(10, 10)


#out, cache = affine_sigmoid_affine_forward(x, w1, b1, w2, b2)
#dx, dw1, db1, dw2, db2 = affine_sigmoid_affine_backward(dout, cache)
#dx_num = eval_numerical_gradient_array(lambda x: 
#                    affine_sigmoid_affine_forward(x, w1, b1, w2, b2)[0], x, dout)
    
out, cache = sigmoid_forward(x)
dx = sigmoid_backward(dout1, cache)
dx_num = eval_numerical_gradient_array(lambda x: sigmoid_forward(x)[0], x, dout1)
    

print('\ndx_num: ', dx_num)
print('\ndx: ', dx)

print('Testing sigmoid_backward function:')
print('dx error: ', rel_meanError(dx_num, dx))
#%%
#===================================================


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
    print('%s relative error: %.2e' % (name, rel_meanError(grad_num, grads[name])))
    
    
#%%

