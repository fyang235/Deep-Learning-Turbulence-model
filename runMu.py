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

#------------------------------------------------------------------------------
#%%
""" read data and subsampling """
loc = r'./instances0.2'
data, mean, std, col_names = prepareData(loc, debug = True)
showData = True
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
    
#learning_rate  = 10**np.linspace(-3, -1, num = 4)
#regularization = 10**np.linspace(-7, -4, num = 4)
#weight_scale   = 10**np.linspace(-6, -3, num = 4)
learning_rate  = [10**(-2.5)] 
regularization = [10**(-7)]
#weight_scale   = [10**(-5)] 
   
#learning_rate  = [10**(-3)] 
#regularization = [10**(-6)]
weight_scale   = [10**(-5)] 
 
D = data['X_train'].shape[1]  
results = {}

LR  = []
REG = []
WS  = []
TEr = []
VEr = []

minErr = np.inf
best_net =None
best_scores = None
best_para = []
for ws in weight_scale:
    for learnrate in learning_rate:
        for reg in regularization:
            model = FullyConnectedNet([500,200,100,50], input_dim=D, num_classes=1,
                             dropout=0.5, use_batchnorm=True, reg=reg,
                             weight_scale=ws, dtype=np.float32, seed=None)
            solver = Solver(model, data,
                            print_every=1000, num_epochs=10, batch_size=1000,
                            update_rule='adam',# sgd, sgd_momentum, rmsprop, adam
                            optim_config={
                              'learning_rate': learnrate,
                            }                       
                     )
            solver.train()
            
            _, y_train_err = solver.check_accuracy(data['X_train'], data['y_train'])
            scores, y_val_err = solver.check_accuracy(data['X_val'], data['y_val'])
            
            results[(learnrate, reg, ws)] = (y_train_err, y_val_err)
            LR.append(np.log10(learnrate))
            REG.append(np.log10(reg))
            WS.append(np.log10(ws))
            TEr.append(y_train_err)
            VEr.append(y_val_err)            
            if y_val_err < minErr:
                minErr = y_val_err
                best_net = solver
                best_scores = scores
                best_para = [learnrate, reg, ws]
            
solver = best_net  
for learnrate, reg, ws in sorted(results):
    train_err, val_err = results[(learnrate, reg, ws)]
    print('lr %e reg %e ws %e train err: %f val err: %f' % (learnrate, reg, ws, train_err, val_err))
print('Best lr: %e, reg: %e, ws: %e, minErr: %f' % (best_para[0], best_para[1], best_para[2], minErr))
print('Exp val: lr: %f, reg: %f, ws: %f' % (np.log10(best_para[0]), np.log10(best_para[1]), np.log10(best_para[2])))
fig1 = plt.figure(1)  
plt.subplot(2,3,1)
plt.plot(LR, TEr, 'o'); plt.xlabel('LR'); plt.ylabel('TEr')
plt.subplot(2,3,2)
plt.plot(REG, TEr, 'o'); plt.xlabel('REG'); plt.ylabel('TEr')
plt.subplot(2,3,3)
plt.plot(WS, TEr, 'o'); plt.xlabel('WS'); plt.ylabel('TEr')
plt.subplot(2,3,4)
plt.plot(LR, VEr, 'o'); plt.xlabel('LR'); plt.ylabel('VEr')
plt.subplot(2,3,5)
plt.plot(REG, VEr, 'o'); plt.xlabel('REG'); plt.ylabel('VEr')
plt.subplot(2,3,6)
plt.plot(WS, VEr, 'o'); plt.xlabel('WS'); plt.ylabel('VEr')
plt.tight_layout()
#%%
#--------------------prediction plot-------------------------------------------
X_train  = data['X_train']
y_train  = data['y_train']
X_val    = data['X_val']
y_val    = data['y_val']
X_test   = data['X_test']
y_test   = data['y_test']
        
#fig = plt.figure(1)
#ax = fig.gca(projection='3d')
#ax.scatter(X_val[:, 0],  X_val[:, 1],  y_val,          label='true value')
#ax.scatter(X_val[:, 0],  X_val[:, 1],  best_scores,  label='prediction curve')
#
#ax.legend()
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')

fig0 = plt.figure(0)

for i in range(X_val.shape[1]):
    plt.subplot(5, 5, i+1)
    plt.plot(X_val[:, i], y_val, 'o')
    plt.plot(X_val[:, i], best_scores, 'o')
    plt.xlabel(col_names[i])
    plt.ylabel(col_names[-1])
plt.tight_layout()

fig2 = plt.figure(2)
#plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')

#test
fig3 = plt.figure(3)

##testset test-----------------
y_pred, err = solver.check_accuracy(data['X_test'], data['y_test'],  batch_size=20000)

plt.subplot(2, 1, 1)
plt.plot(y_test, y_pred, 'o')
maxval = np.maximum(np.max(y_test), np.max(y_pred))
minval = np.minimum(np.min(y_test), np.min(y_pred))
plt.plot([minval,maxval], [minval,maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')

y_pred_org = y_pred*std[-1] + mean[-1]
y_test_org = data['y_test']*std[-1] + mean[-1]
plt.subplot(2, 1, 2)
plt.plot(y_test_org, y_pred_org, 'o')
maxval = np.maximum(np.max(y_pred_org), np.max(y_test_org))
minval = np.minimum(np.min(y_pred_org), np.min(y_test_org))
plt.plot([minval,maxval], [minval,maxval])
plt.xlabel('y_test_org')
plt.ylabel('y_pred_org')

plt.tight_layout()
# plt.show()

import os
pwd = os.path.abspath('.')
image_dir = os.path.join(pwd, 'Images')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

fig0.savefig('./Images/fig0')
fig1.savefig('./Images/fig1')
fig2.savefig('./Images/fig2')
fig3.savefig('./Images/fig3')

#debug test-----------------
#Xtest5 = np.array([
#[4.10940e-04,	-5.90212e-02,	4.52778e-01,		2.00000e+00,		-4.19455e-01,	6.92636e-01,		3.28721e-01, 	4.64430e-01, 	4.37134e-01 ,	5.54858e-01 	], 
#[4.10940e-04,	8.04822e-03	,	5.42853e-01,		2.00000e+00,		-4.98739e-01,	6.92590e-01,		2.87111e-01, 	3.19531e+00, 	-2.59869e-01, 	5.52231e-01 	],
#[4.10940e-04,	1.28574e-02 ,	6.37210e-01,		2.00000e+00,		-4.89379e-01,	6.92583e-01,		2.44475e-01, 	4.97963e+32, 	-2.95943e-01, 	5.53333e-01 	],
#[4.10940e-04,	3.36433e-03 ,	6.85605e-01,		1.69284e+00,		-3.97526e-01,	6.92582e-01,		2.12345e-01, 	1.22713e+32, 	-2.49301e-01, 	5.53121e-01 	],
#[4.10940e-04,	-2.93323e-03, 	6.88469e-01,		1.50557e+00,		-2.94183e-01,	6.92581e-01,		1.91714e-01, 	5.97641e+31, 	-1.83490e-01, 	5.52992e-01 	]
#])
#ytest5_org = np.array([
#4.89693e-04,
#4.02742e-04,
#3.23584e-04,
#2.69592e-04,
#2.37185e-04])
#Xtest5 = (Xtest5 - mean[:10] )
#Xtest5 = Xtest5/std[:10];
#ytest5 = (ytest5_org - mean[-1] )/std[-1];

#y_pred, err = solver.check_accuracy(Xtest5, ytest5,  batch_size=1000)
#plt.subplot(2, 1, 1)
#plt.plot(ytest5, y_pred, 'o')
#maxval = np.maximum(np.max(ytest5), np.max(y_pred))
#minval = np.maximum(np.min(ytest5), np.min(y_pred))
#plt.plot([minval,maxval], [minval,maxval])
#plt.xlabel('ytest5')
#plt.ylabel('y_pred')
#
#y_pred_org = y_pred*std[-1] + mean[-1]
#plt.subplot(2, 1, 2)
#plt.plot(ytest5_org, y_pred_org, 'o')
#maxval = np.maximum(np.max(y_pred_org), np.max(ytest5_org))
#minval = np.maximum(np.min(y_pred_org), np.min(ytest5_org))
#plt.plot([minval,maxval], [minval,maxval])
#plt.xlabel('ytest5_org')
#plt.ylabel('y_pred_org')
#plt.show()


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
#%%
#----output files for openfoam reading----
outPutDataSet = solver.best_params
outPutDataSet['mean'] = mean
outPutDataSet['std'] = std

parameter_dir = os.path.join(pwd, 'MLparams')
if not os.path.exists(parameter_dir):
    os.mkdir(parameter_dir)

for k, V in outPutDataSet.items():
    if k[:4] == 'mode':
        continue
    with open('MLparams/'+ str(k), 'w') as f:
        f.write(
r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.7.1                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\/     M anipulation  |                                                 |
\*----------------------------------------------------------------------------/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      MLparams;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
""")

        # output 2 d tensor
        if len(V.shape) == 2:
            f.write('\n//%s RectangularMatrix <doubleScalar>  \n' % k)
            f.write('%d %d \n' % (V.shape[0], V.shape[1]))
            f.write('(\n')
            for w in V:
                f.write('( ')
                for n in w:
                    f.write('%15.5e' % n)
                f.write(' )\n')
            f.write(') \n')
        # output 1 d tensor
        elif len(V.shape) == 1:
            f.write('\n//%s RectangularMatrix <doubleScalar> \n' % k)
            f.write('%d %d\n' % (V.shape[0], 1))
            f.write('(\n')
            for w in V:
                f.write('( ')
                f.write('%15.5e' % w)
                f.write(' ) \n')
            f.write(') \n')
        else:
            print("Streng data shape!")
        

        
