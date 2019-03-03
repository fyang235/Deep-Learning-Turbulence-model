import numpy as np
from layers import *

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=5, num_classes=1,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
     
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

   #initialize
        
        for i in range(1, self.num_layers + 1):       
            if i == 1:
                self.params['W'+str(i)] = weight_scale*np.random.randn(input_dim, hidden_dims[i-1])
                self.params['b'+str(i)] = np.zeros(hidden_dims[i-1])   
                if self.use_batchnorm:
                    self.params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
                    self.params['beta'+str(i)] = np.zeros(hidden_dims[i-1])  
                    
            if 1 < i and i < self.num_layers:                  
                self.params['W'+str(i)] = weight_scale*np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
                self.params['b'+str(i)] = np.zeros(hidden_dims[i-1])
                if self.use_batchnorm:
                    self.params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
                    self.params['beta'+str(i)] = np.zeros(hidden_dims[i-1]) 
                    
            if i == self.num_layers:
                self.params['W'+str(i)] = weight_scale*np.random.randn(hidden_dims[-1], num_classes)
                self.params['b'+str(i)] = np.zeros(num_classes)               

#            print('W{}: '.format(i), self.params['W'+str(i)].shape)
#            print('b{}: '.format(i), self.params['b'+str(i)].shape)
#            if i != self.num_layers:
#                print('gamma{}: '.format(i), self.params['gamma'+str(i)].shape)
#                print('beta{}: '.format(i), self.params['beta'+str(i)].shape)        


        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
 
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        
 #forward
        cache_list = []
        ipt = X
        for i in range(1, self.num_layers):
            cache = []
            ipt, cache_affine = affine_forward(ipt, self.params['W'+str(i)], self.params['b'+str(i)])
            cache.append(cache_affine)
#------------------------------------------------------------------------------            
            if self.use_batchnorm:
                ipt, cache_bn = batchnorm_forward(ipt, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
                cache.append(cache_bn)
#------------------------------------------------------------------------------                
            ipt, cache_relu = relu_forward(ipt)
            cache.append(cache_relu)
#------------------------------------------------------------------------------            
            if self.use_dropout:
                ipt, cache_drop = dropout_forward(ipt, self.dropout_param)
                cache.append(cache_drop)
            cache_list.append(cache)
            
        scores, cache_affine = affine_forward(ipt, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        cache_list.append(cache_affine)
#------------------------------------------------------------------------------
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
 #backward
        #get loss

        loss, dscores = square_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            W = self.params['W'+str(i)]
            loss += 0.5*self.reg*np.sum(W**2)
            
        cache = cache_list[-1]
        
        dipt, dW, db = affine_backward(dscores, cache)
        dW += self.reg * self.params['W' + str(self.num_layers)]
        grads['W'+str(self.num_layers)] = dW
        grads['b'+str(self.num_layers)] = db
        
        for i in reversed(range(1, self.num_layers)):     
            W = self.params['W'+str(i)]
            cache = cache_list[i-1]
            
            k = -1
            if self.use_dropout:
                dipt = dropout_backward(dipt, cache[k])
                k -= 1
                
            dtemp = relu_backward(dipt, cache[k])
            k -= 1
            
            if self.use_batchnorm:
                dtemp, dgamma, dbeta = batchnorm_backward(dtemp, cache[k])
                grads['gamma'+str(i)] = dgamma
                grads['beta'+str(i)] = dbeta       
                k -= 1
                
            dipt, dW, db = affine_backward(dtemp, cache[k])  
            dW += self.reg*W
            grads['W'+str(i)] = dW
            grads['b'+str(i)] = db            

        return loss, grads