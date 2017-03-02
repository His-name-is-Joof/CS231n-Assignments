import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
    WW = (input_dim[2] - filter_size + 2 * ((filter_size - 1) / 2)) / 1 + 1
    HH = (input_dim[1] - filter_size + 2 * ((filter_size - 1) / 2)) / 1 + 1
    WWW = (WW - 2) / 2 + 1
    HHH = (HH - 2) / 2 + 1
    layer_size = WWW * HHH * num_filters
    self.params['W2'] = np.random.normal(0, weight_scale, (layer_size, hidden_dim))
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    reg = self.reg
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    # intermediates
    conv_cache  = []
    conv_out    = []
    relu_cache  = []
    relu_out    = []
    pool_cache  = []
    pool_out    = []
    affine_cache= []
    affine_out  = []
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    o, c = conv_forward_fast(X, W1, b1, conv_param)
    #o, c = conv_forward_naive(X, W1, b1, conv_param)
    conv_cache  += [c]
    conv_out    += [o]
    o, c = relu_forward(o)
    relu_cache  += [c]
    relu_out    += [o]
    o, c = max_pool_forward_fast(o, pool_param)
    pool_cache  += [c]
    pool_out    += [o]

    o, c = affine_relu_forward(o, W2, b2)
    affine_cache+= [c]
    affine_out  += [o]

    o, c = affine_forward(o, W3, b3)
    affine_cache+= [c]
    affine_out  += [o]

    scores = o
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    for idx in xrange(3):
        weight_str = 'W' + str(idx + 1)
        weight = self.params[weight_str]
        loss += 0.5 * reg * np.sum(weight * weight)

    weight_str  = 'W3'
    bias_str    = 'b3'
    weight = self.params[weight_str]
    dscores, dW, db = affine_backward(dscores, affine_cache[1])
    grads[weight_str]   = dW + reg * weight
    grads[bias_str]     = db

    weight_str  = 'W2'
    bias_str    = 'b2'
    weight = self.params[weight_str]
    dscores, dW, db = affine_relu_backward(dscores, affine_cache[0])
    grads[weight_str]   = dW + reg * weight
    grads[bias_str]     = db

    weight_str  = 'W1'
    bias_str    = 'b1'
    weight = self.params[weight_str]
    dscores = max_pool_backward_fast(dscores, pool_cache[0])
    dscores = relu_backward(dscores, relu_cache[0])
    dscores, dW, db = conv_backward_fast(dscores, conv_cache[0])

    grads[weight_str]   = dW + reg * weight
    grads[bias_str]     = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
