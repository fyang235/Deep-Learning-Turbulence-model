import numpy as np
#==========

def affine_sigmoid_affine_squareloss_forward(x, w1, b1, w2, b2, y):
    loss, out, cache = None, None, None
    
    out1, cache1 = affine_forward(x, w1, b1)
    out2, cache2 = sigmoid_forward(out1)
    out3, cache3 = affine_forward(out2, w2, b2)
    loss, dout3 = square_loss(out3, y)
    
    cache = (cache1, cache2, cache3, dout3)
    out = out3
    return loss, out, cache

def affine_sigmoid_affine_squareloss_backward(cache):
    dx, dw1, db1, dw2, db2 = None, None, None, None, None  
    cache1, cache2, cache3, dout3 = cache
    
    dout2, dw2, db2 = affine_backward(dout3, cache3)
    dout1 = sigmoid_backward(dout2, cache2)
    dx, dw1, db1 = affine_backward(dout1, cache1)
    
    return dx, dw1, db1, dw2, db2

#==========
def affine_sigmoid_affine_forward(x, w1, b1, w2, b2):
    out, cache = None, None
    
    out1, cache1 = affine_forward(x, w1, b1)
    out2, cache2 = sigmoid_forward(out1)
    out3, cache3 = affine_forward(out2, w2, b2)
    
    cache = (cache1, cache2, cache3)
    out = out3
    return out, cache

def affine_sigmoid_affine_backward(dout, cache):
    dx, dw1, db1, dw2, db2 = None, None, None, None, None  
    cache1, cache2, cache3 = cache
    
    dout2, dw2, db2 = affine_backward(dout, cache3)
    dout1 = sigmoid_backward(dout2, cache2)
    dx, dw1, db1 = affine_backward(dout1, cache1)
    
    return dx, dw1, db1, dw2, db2

#==========
def affine_sigmoid_forward(x, w, b):
    
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = sigmoid_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_sigmoid_backward(dout, cache):

    fc_cache, relu_cache = cache
    da = sigmoid_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db 
#==========
def affine_relu_forward(x, w, b):

    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db    
#==========
#------------------------------------------------------------------------------

def affine_forward(x, w, b, debug = False):
    out, cache = None, None
    
    x_row = np.reshape(x, (x.shape[0], -1))
    out = np.dot(x_row, w) + b
    
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache, debug = False):
    dx, dw, db = None, None, None
    x, w, b = cache

    x_row = np.reshape(x, (x.shape[0], -1))   

    dx = np.dot(dout, w.T)
    dw = np.dot(x_row.T, dout)
    db = np.sum(dout, axis = 0)    
    dx = dx.reshape(x.shape)    

    return dx, dw, db
#------------------------------------------------------------------------------
def sigmoid_forward(x, debug = False):
    out, cache = None, None
    
    out = 1/(1 + np.exp(-x))
    
    cache = x
    return out, cache

def sigmoid_backward(dout, cache, debug = False):
    dx = None
    x  = cache
    
    sig, _ = sigmoid_forward(x)
    dx = dout * sig * (1 - sig)

    return dx
#------------------------------------------------------------------------------
def relu_forward(x):
    out = None

    mask = np.array(x > 0, dtype = float)
    out = x*mask

    cache = x
    return out, cache

def relu_backward(dout, cache):

    dx, x = None, cache

    dx = dout*np.array(x > 0, dtype = float)
    return dx
#------------------------------------------------------------------------------

def batchnorm_forward(x, gamma, beta, bn_param, debug = False):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        
        mu = np.mean(x, axis = 0)
        xmu = x - mu
        sqxmu = xmu**2
        var = np.mean(sqxmu, axis = 0)
        sqrtvar = np.sqrt(var + eps)
        ivar = 1/sqrtvar
        xhat = xmu * ivar
        gammaxhat = gamma * xhat
        out = gammaxhat + beta
        
        cache = (mu, xmu, sqxmu, var, sqrtvar, ivar, xhat, gammaxhat, out, eps, gamma, beta, x)
        
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var  = momentum * running_var  + (1 - momentum) * var        
        
    elif mode == 'test':           
        x_bn = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma*x_bn + beta
       
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batchnorm_backward(dout, cache, debug = False):
    
    dx, dgamma, dbeta = None, None, None
   
    N, D = dout.shape
    mu, xmu, sqxmu, var, sqrtvar, ivar, xhat, gammaxhat, out, eps, gamma, beta, x = cache
    
    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(xhat*dout, axis = 0)

    dxhat = dout*gamma
    dx = (N*dxhat - np.sum(dxhat, axis = 0) - xhat*np.sum(dxhat*xhat, axis = 0))/N*ivar

    return dx, dgamma, dbeta
#------------------------------------------------------------------------------
    
def dropout_forward(x, dropout_param, debug = False):

    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':

        mask = (np.random.rand(*x.shape) < (1-p)) / (1-p)
        out = x * mask

    elif mode == 'test':

        out = x 

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache, debug = False):

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
 
        dx = mask * dout
        
    elif mode == 'test':
        dx = dout
    return dx
#------------------------------------------------------------------------------
    
def square_loss(x, y, debug = False):
    loss, dx = None, None
    
    #for regression we need to reshape y as x.shape
    if x.shape[1] == 1: y = y.reshape(*x.shape)
    loss = np.mean(0.5 * (x - y)**2)
    dx = (x - y) / len(x)
#    print('(x - y)**2: ', (x - y)**2)
#    print('x: ', x)
#    print('y: ', y)
#    print('dx: ', dx)
    return loss, dx

def crossEntropy_loss(x, y, debug = False):
    loss, dx = None, None
    
    #for regression we need to reshape y as x.shape
    if x.shape[1] == 1: y = y.reshape(*x.shape)
    eps = 1e-8
    loss =  np.mean(-y * np.log(np.maximum(x, eps))-(1-y) * np.log(np.maximum(1-x, eps)))
    dx = (x - y.reshape(*x.shape)) / (x * (1 - x)) / len(x)
    
    return loss, dx