import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0

  max_func = np.zeros((500, 10))

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    nonzeros = 0
    correct_class_x = 0
    correct_class_y = 0
    for j in xrange(num_classes):
      if j == y[i]:
          correct_class_x = i
          correct_class_y = j
          continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        max_func[i, j] = 1
        nonzeros += 1
        loss += margin
    max_func[correct_class_x, correct_class_y] = -1 * nonzeros

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss /= num_train

  # Add regularization to the loss
  regularization_constant = 0.5 * reg * np.sum(W * W)
  loss += regularization_constant


  dW = max_func
  dW = X.T.dot(dW)
  dW = np.divide(dW, num_train)
  dW = np.add(dW, reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  class_scores = scores[np.arange(y.size), y]  # Wc
  class_scores = class_scores * -1             # -Wc
  class_scores = class_scores + 1              # -Wc + 1
  margins = scores.T + class_scores            #Wi - Wc + 1
  margins = margins.T
  margins[margins < 0] = 0
  loss = margins
  loss[np.arange(y.size), y] = 0
  loss = loss.sum()
  loss /= X.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dmargins = margins
  dmargins[np.arange(y.size), y] = 0
  dmargins[margins >0] = 1 
  dmargins_sum = dmargins.sum(1)
  dmargins_sum = dmargins_sum * -1
  dmargins[np.arange(y.size), y] = dmargins_sum
  dscores = X.T.dot(dmargins)
  dscores = np.divide(dscores, X.shape[0])
  dW = dscores + (reg * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
