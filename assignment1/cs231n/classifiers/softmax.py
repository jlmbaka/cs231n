import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[1]
  num_classes = W.shape[0]
  #############################################################################
  #       Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train): # for each image
    # compute the score
    scores = W.dot(X[:, i])

    # shift the values of f so that the highest number is 0:
    scores -= np.max(scores)

    # compute the loss
    loss += -np.log(np.exp(scores[y[i]]) / np.sum(np.exp(scores)))

    # gradient(https://github.com/seyedamo/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py)
    scores = np.exp(scores)
    scores /= np.sum(scores)
    for j in range(num_classes): # for each class
      dW[j, :] += scores[j] * X[:, i].T

    # dW wrt correct class scores w_yi
    dW[y[i], :] += -X[:, i].T

  # Average the loss 
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # average of the gradient
  dW /= num_train
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
  num_train = X.shape[1]
  num_classes = W.shape[0]
  #############################################################################
  #  Compute the softmax loss and its gradient using no explicit loops.       #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # compute scores
  scores = W.dot(X)
  scores -= np.max(scores)

  # softmax function
  softmax = np.exp(scores) / np.sum(np.exp(scores), 0) # 10 x 49000 | C x D
  
  # cross entropy loss
  loss = -np.log(softmax[y, range(num_train)]) # 49000
  loss = np.sum(loss) / num_train

  # regularisation
  loss += 0.5 * reg * np.sum(W*W)

  # gradient (source:https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py)
  ind = np.zeros(softmax.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((softmax-ind), X.T)
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
