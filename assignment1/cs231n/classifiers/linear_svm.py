import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in range(num_train): # for each training image
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]

    count_of_classes_that_didnt_meet_the_expected_margin = 0

    for j in range(num_classes): # for each class j
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if margin > 0:
        # Update the loss function
        loss += margin

        # Update the count of classes that didn't meet the expected margin (> 0)
        count_of_classes_that_didnt_meet_the_expected_margin += 1

        # Update the gradient for other rows where j != y_i [gradient formula 2]
        # *Explantion of why we update it inside the if statement* :
        #   the indicator function returns 1 if the condition inside the () is true (=margin > 0)
        #   or zero otherwise.
        #   Therefore, we only update it when margin > 0 and leave it to 0 in the other case
        dW[j, :] += X[:, i]

    # Gradient wrt to the row of W that corresponds to the correct class:
    # Count the number of classes that didn't meet the desired margin and then
    # the data vector X_i multiplied by this number is the gradient
    #p rint(count_of_classes_that_didnt_meet_the_expected_margin)
    dW[y[i], :] += -count_of_classes_that_didnt_meet_the_expected_margin * X[:, i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # average of the gradient
  dW /= num_train

  #############################################################################
  # Compute the gradient of the loss function and store it in dW.             #
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
  N = len(y) # number of training data aka num_train

  # compute scores
  scores = W.dot(X) # C x N

  # compute the margins for all the classes and images in one vector operation
  correct_class_score = scores[y, range(N)]
  margins = np.maximum(0, scores - correct_class_score + 1) # delta = 1

  # on y-th positions scores[y] - score[y, range(N)] canceled and gave delta. We
  # Want to ignore the y-th positions and only consider margins on max wrong class
  margins[y, range(N)] = 0

  # We only compute one sum because our vector operation has already computed the
  # score for each image..
  loss = np.sum(margins) / N

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

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

  select_wrong = np.zeros(margins.shape)
  select_wrong[margins > 0] = 1 # wrong margins

  select_correct = np.zeros(margins.shape)
  select_correct[y, range(N)] = np.sum(select_wrong, axis=0) # correct margins

  dW = select_wrong.dot(X.T)
  dW -= select_correct.dot(X.T)
  dW /= N

  # add regularisation gradient
  # dW += reg * W

  # [dw] vectorised count of classes that didn't meet the desired margin per image
  # no_unexpected_margin = np.sum(margins > 0, 0)
  # idx_didnt_meet_expected_margin = margins > 0

  # dw[no_unexpected_margin > 0, :] = no_unexpected_margin * X[no_unexpected_margin > 0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
