import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.]
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = y.size
  num_classes = W.shape[1]

  intermediate = np.zeros((num_train, num_classes))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    margin = 0
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        score_exp = np.exp(scores[j])
        intermediate[i, j] = score_exp
        margin += score_exp
    intermediate[i, :] /= margin
    for j in xrange(num_classes):
        if j == y[i]:
            intermediate[i, j] -= 1
    margin = np.log(margin)
    margin = np.nan_to_num(margin)
    margin -= correct_class_score
    loss += margin

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  intermediate /= num_train
  dW = X.T.dot(intermediate)
  dW += 0.5 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores_exp = np.exp(scores)
  scores_exp_sum = scores_exp.sum(1)
  scores_correct = scores[np.arange(y.size), y]
  scores_exp_sum_log = np.log(scores_exp_sum)
  scores_exp_sum_log = np.nan_to_num(scores_exp_sum_log)
  loss = -scores_correct + scores_exp_sum_log
  loss = loss.sum()
  loss /= y.size

  regularization = 0.5 * reg * np.sum(W * W)

  loss += regularization

  dscores = np.divide(scores_exp.T, scores_exp_sum)
  dscores = dscores.T
  dscores[np.arange(y.size), y] += -1
  dscores = np.divide(dscores, y.size)

  dW = X.T.dot(dscores)
  dW = dW + (0.5 * reg * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

