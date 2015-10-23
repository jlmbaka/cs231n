# if __name__ == "__main__":
#     # inputs
#     results = {}
#     best_val = -1
#     best_softmax = None
#     learning_rates = [1e-7, 5e-7]
#     regularization_strengths = [5e4, 1e8]
#     no_steps = 10.0
#     verbose = False

#     # run fn
#     tic = time.time()
#     [result, best_val, best_softmax] = tune_learn_rate_and_reg(learning_rates, regularization_strengths, no_steps, verbose=False)
#     toc = time.time()

import numpy as np
from cs231n.classifiers import Softmax
from cs231n.classifiers import LinearSVM

def tune_learn_rate_and_reg(classifier_maker, X_train, y_train, X_val, y_val, learning_rates, regularization_strengths, num_iters=150, verbose=False):
    # outputs
    best_val = -1
    best_softmax = None
    results = {}

    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            if verbose:
                print("**(Learning rate, regularisation strength) = ({}, {})**".format(learning_rate, reg))
            classifier = classifier_maker()
            
            # train
            classifier.train(X_train, y_train, learning_rate, reg, num_iters, verbose=verbose)
            
            # predict
            y_train_pred = classifier.predict(X_train)
            y_val_pred = classifier.predict(X_val)
            
            # compute and set the best validation accuracy and the best svm
            validation_accuracy = np.mean(y_val == y_val_pred)
            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_classifier = classifier
            # compute the training accuracy
            training_accuracy =  np.mean(y_train == y_train_pred)
            
            # store in the results dictionary
            results[(learning_rate, reg)] = (training_accuracy, validation_accuracy)

    return [results, best_val, best_classifier]


def gen_no_steps_in_range(start, end, no_steps):
    return np.array(np.arange(start, end, (end-start)/no_steps))