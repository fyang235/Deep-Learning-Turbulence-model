from data_util import prepareData, splitData
from layers import *
from gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from nnet import FullyConnectedNet
import matplotlib.pyplot as plt
from solver import Solver
import time
from relativeError import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#%%
#generate data
N = 1000 # number of points 
std_variance = 0.1
mixing = 4
t = np.linspace(0, 6*np.pi, N) 

#X1 = np.sin(t) + np.random.randn(N)*std_variance
X2 = np.cos(2*t) + np.random.randn(N)*std_variance
X1 = t**3
#X2 = t
Xd = np.c_[X1, X2]
Yd = t + np.random.randn(N)*std_variance
dataSet = np.c_[Xd, Yd]

#preprocessing
dataSet -= np.mean(dataSet, axis = 0)
dataSet /= np.std(dataSet, axis = 0)
data = splitData(dataSet, p_train = 0.9, p_val = 0.1, debug = False)
#------------------------------------------------------------------------------
#%%
""" read data and subsampling """
loc = r'C:\Research_ERC\CS\machineLearning\ML applications\dataExtracting\instances'
#data = prepareData(loc)
showData = False
if showData:
    for k, v in list(data.items()):
        print('%s shape: ' % k, v.shape)
#------------------------------------------------------------------------------
#%%    
""" test modular layers """
testLayerFcn = False
if testLayerFcn:    
    import testLayers
#------------------------------------------------------------------------------
#%%
""" test modular layers """
testNet = False
if testNet:    
    import testNetGrad
#------------------------------------------------------------------------------
#%%

#learning_rate  = 10**np.linspace(-6, -1, num = 6)
#regularization = 10**np.linspace(-6, -1, num = 6)
#weight_scale   = 10**np.linspace(-6, -1, num = 6)
regularization = [0.1]
learning_rate  = [0.1]  
    
results = {}
minErr = np.inf
best_net =None
best_scores = None

D = data['X_train'].shape[1]  
results = {}
minErr = np.inf
best_net =None
best_scores = None

for ws in weight_scale:
    for learnrate in learning_rate:
        for reg in regularization:
            model = FullyConnectedNet([100, 100], input_dim=D, num_classes=1,
                             dropout=0.5, use_batchnorm=True, reg=reg,
                             weight_scale=ws, dtype=np.float32, seed=None)
            solver = Solver(model, data,
                            print_every=400, num_epochs=50, batch_size=100,
                            update_rule='sgd_momentum',# sgd, sgd_momentum, rmsprop, adam
                            optim_config={
                              'learning_rate': learnrate,
                            }                       
                     )
            solver.train()
            
            _, y_train_err = solver.check_accuracy(data['X_train'], data['y_train'])
            scores, y_val_err = solver.check_accuracy(data['X_val'], data['y_val'])
            
            results[(learnrate, reg, ws)] = (y_train_err, y_val_err)
            if y_val_err < minErr:
                minErr = y_val_err
                best_net = solver
                best_scores = scores
            
solver = best_net  
for learnrate, reg, ws in sorted(results):
    train_err, val_err = results[(learnrate, reg, ws)]
    print('lr %e reg %e ws %e train err: %f val err: %f' % (learnrate, reg, ws, train_err, val_err))
print('minErr achieved during cross-validation: %f' % minErr)
  


#%%
#--------------------prediction plot-------------------------------------------
X_train  = data['X_train']
y_train  = data['y_train']
X_val    = data['X_val']
y_val    = data['y_val']
X_test   = data['X_test']
y_test   = data['y_test']
        
fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.scatter(X_val[:, 0],  X_val[:, 1],  y_val,          label='true value')
ax.scatter(X_val[:, 0],  X_val[:, 1],  best_scores,  label='prediction curve')

ax.legend()
ax.set_xlabel('X1')
ax.set_ylabel('X2')

plt.figure(0)
for i in range(X_val.shape[1]):
    plt.subplot(1, 2, i+1)
    plt.plot(X_val[:, i], y_val, 'o')
    plt.plot(X_val[:, i], best_scores, 'o')

plt.figure(2)
#plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')

#test
plt.figure(3)
y_pred, err = solver.check_accuracy(data['X_test'], data['y_test'],  batch_size=1000)
#plt.subplot(2, 1, 2)
plt.plot(y_test, y_pred, 'o')
maxval = np.maximum(np.max(y_test), np.max(y_test))
minval = np.maximum(np.min(y_test), np.min(y_test))
plt.plot([minval,maxval], [minval,maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()
#------------------------------------------------------------------------------
#%%
""" sgd vs sgd_moment"""
#num_train = 500
#small_data = {
#  'X_train': data['X_train'][:num_train],
#  'y_train': data['y_train'][:num_train],
#  'X_val': data['X_val'],
#  'y_val': data['y_val'],
#}
#
#solvers = {}
#
#for update_rule in ['sgd', 'sgd_momentum']:
#  print('running with ', update_rule)
#  model = FullyConnectedNet([10], weight_scale=5e-2)
#
#  solver = Solver(model, small_data,
#                  num_epochs=5, batch_size=100,
#                  update_rule=update_rule,
#                  optim_config={
#                    'learning_rate': 1e-2,
#                  },
#                  verbose=True)
#  solvers[update_rule] = solver
#  solver.train()
#  print()
#
#plt.subplot(3, 1, 1)
#plt.title('Training loss')
#plt.xlabel('Iteration')
#
#plt.subplot(3, 1, 2)
#plt.title('Training accuracy')
#plt.xlabel('Epoch')
#
#plt.subplot(3, 1, 3)
#plt.title('Validation accuracy')
#plt.xlabel('Epoch')
#
#for update_rule, solver in list(solvers.items()):
#  plt.subplot(3, 1, 1)
#  plt.plot(solver.loss_history, 'o', label=update_rule)
#  
#  plt.subplot(3, 1, 2)
#  plt.plot(solver.train_acc_history, '-o', label=update_rule)
#
#  plt.subplot(3, 1, 3)
#  plt.plot(solver.val_acc_history, '-o', label=update_rule)
#  
#for i in [1, 2, 3]:
#  plt.subplot(3, 1, i)
#  plt.legend(loc='upper center', ncol=4)
#plt.gcf().set_size_inches(15, 15)
#plt.show()
#%%
#learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3}
#for update_rule in ['adam', 'rmsprop']:
#  print('running with ', update_rule)
#  model = FullyConnectedNet([4], weight_scale=5e-2)
#
#  solver = Solver(model, data,
#                  num_epochs=10, batch_size=100,
#                  update_rule=update_rule,
#                  optim_config={
#                    'learning_rate': learning_rates[update_rule]
#                  },
#                  verbose=True)
#  solvers[update_rule] = solver
#  solver.train()
#  print()
#
#plt.subplot(3, 1, 1)
#plt.title('Training loss')
#plt.xlabel('Iteration')
#
#plt.subplot(3, 1, 2)
#plt.title('Training accuracy')
#plt.xlabel('Epoch')
#
#plt.subplot(3, 1, 3)
#plt.title('Validation accuracy')
#plt.xlabel('Epoch')
#
#for update_rule, solver in list(solvers.items()):
#  plt.subplot(3, 1, 1)
#  plt.plot(solver.loss_history, 'o', label=update_rule)
#  
#  plt.subplot(3, 1, 2)
#  plt.plot(solver.train_acc_history, '-o', label=update_rule)
#
#  plt.subplot(3, 1, 3)
#  plt.plot(solver.val_acc_history, '-o', label=update_rule)
#  
#for i in [1, 2, 3]:
#  plt.subplot(3, 1, i)
#  plt.legend(loc='upper center', ncol=4)
#plt.gcf().set_size_inches(15, 15)
#plt.show()