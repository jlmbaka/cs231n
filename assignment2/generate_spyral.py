import numpy as np
import matplotlib.pyplot as plt

# visualise dataset
N = 200 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# Training a softmax linear classifier

# initialise parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength


# gradient descent loop
num_examples = X.shape[0]
print(num_examples)

for i in range(200):

	# complute the class score
	scores = np.dot(X, W) + b

	# compute the loss
	## get unnormalised probalilities
	exp_scores = np.exp(scores)
	## normalise them for each example
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	## probabilities assigned to the correct classes in each example
	correct_log_probs = -np.log(probs[range(num_examples),list(y)])
	## compute the loss: average cross-entropy loss and regularisation loss
	data_loss = np.sum(correct_log_probs)/num_examples
	reg_loss = 0.5*reg*np.sum(W*W)
	loss = data_loss + reg_loss
	if i % 10 == 0:
		print("iteration {:d} loss {:f}".format(i, loss))

	# computing the analytic gradient with backpropagation
	dscores = probs
	dscores[range(num_examples), y] -= 1
	dscores /= num_examples

	## backpropagate into W and b
	dW = np.dot(X.T, dscores)
	db = np.sum(dscores, axis=0, keepdims=True)
	dW += reg*W # don't forget the regularisation gradient

	# Performing a parameter update
	W += -step_size * dW
	b += -step_size * db


# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: {:2f}".format(np.mean(predicted_class == y)))



# TRAINING A NEURAL NETWORK
## initialise parameters randomly
h = 100 # size of hidden layer
W = np.random.randn(D,h) * np.sqrt(2.0/(D*h))
b = np.zeros((1,h))

W1 = np.random.randn(h,h) * np.sqrt(2.0/(h*h))
b1 = np.zeros((1,h))

W2 = np.random.randn(h,K) * np.sqrt(2.0/(h*K))
b2 = np.zeros((1,K))

p = 0.5
for i in range(10000):

	# forward pass to compute scores
	## evaluate class scores with a 3 layer NN

	hidden_layer = np.maximum(0, np.dot(X,W) + b) # note, ReLU activation
	### dropout hidden_layer ###
	U = (np.random.rand(*hidden_layer.shape) < p) / p # first dropout mask
	hidden_layer *= U # drop!
	### dropout ###

	hidden_layer1 = np.maximum(0, np.dot(hidden_layer, W1) + b1)
	### dropout hidden_layer ###
	U1 = (np.random.rand(*hidden_layer1.shape) < p) / p # first dropout mask
	hidden_layer1 *= U1 # drop!
	### dropout ###

	# output
	scores = np.dot(hidden_layer1, W2) + b2

	# compute the class probabilities
	exp_scores = np.exp(scores)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

	# compute the loss: average cross-entropy loss and regularization
	corect_logprobs = -np.log(probs[range(num_examples),y])
	data_loss = np.sum(corect_logprobs)/num_examples
	reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
	loss = data_loss + reg_loss
	if i % 1000 == 0:
		print("iteration %d: loss %f" % (i, loss))

	# compute the gradient on scores
	dscores = probs
	dscores[range(num_examples),y] -= 1
	dscores /= num_examples


	# backpropagate the gradient
	## first backprop into parameters W2 and b2
	dW2 = np.dot(hidden_layer1.T, dscores)
	db2 = np.sum(dscores, axis=0, keepdims=True)
	## backprop into hidden layer 1
	dhidden_layer1 = np.dot(dscores, W2.T)
	## backprop the ReLu non-linearity
	dhidden_layer1[hidden_layer1 <= 0] = 0

	## second backrop into parameters w1 and b1
	dW1 = np.dot(hidden_layer.T, dhidden_layer1)
	db1 = np.sum(dhidden_layer1, axis=0, keepdims=True)
	## backrpop in hidden layer
	dhidden_layer = np.dot(dhidden_layer1,  W1.T)
	## backprop into ReLu non-linearity
	dhidden_layer[hidden_layer <= 0] = 0


	# finally into W, b
	dW = np.dot(X.T, dhidden_layer)
	db = np.sum(dhidden_layer, axis=0, keepdims=True)

	# add regularization gradient contribution
	dW2 += reg * W2
	dW1 += reg * W1
	dW 	+= reg * W

	# perform a parameter update
	W += -step_size * dW
	b += -step_size * db
	W1 += -step_size * dW1
	b1 += -step_size * db1
	W2 += -step_size * dW2
	b2 += -step_size * db2


# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
hidden_layer1 = np.maximum(0, np.dot(hidden_layer, W1) + b1)
scores = np.dot(hidden_layer1, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))






