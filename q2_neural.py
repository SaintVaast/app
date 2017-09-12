#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # raise NotImplementedError
    x = data # M x Dx
    x2 = np.dot(x,W1)+b1 # M x Dh
    h1 = sigmoid(x2) # M x Dh
    h2 = np.dot(h1,W2)+b2 # M x Dy
    yhat = softmax(h2) # M x Dy
    def CE(y,yhat):
        """
        Computes the cross-entropy between true and predicted labels.
        Arguments :
        y -- Matrix or vector of one-hot-encoded true labels
             One row is one label
             M x Dy
        yhat -- Matrix or vector of probabilities output by the nn.
                One row is one vector of probabilities.
                M x Dy
        Return :
        J -- Cross-entropy cost.
        """
        lyhat = np.log(yhat)
        elmtwsprdct = (y*lyhat).T # each column of this matrix is the elementwise prod\
        # between the true one-hot label and the log of the predicted one.
        J = np.sum(elmtwsprdct,axis=0).T # 1 x M
        J = J.mean()
        return(J)
    y = labels # M x Dy
    cost = CE(y,yhat) # vector of costs, M x 1
    # cost = cost.mean(axis=0)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # raise NotImplementedError    
    dh2 = yhat
    dh2[labels==1] += -1 # M x Dy
    dh2 /= -len(data)
    dW2 = h1.T.dot(dh2) # Dh x Dy
    db2 = np.sum(dh2, axis=0) # Dy x 1
    dh1 = dh2.dot(W2.T) # M x Dh
    dx2 = dh1*sigmoid_grad(h1) # M x Dh
    dW1 = x.T.dot(dx2) # Dx x Dh
    db1 = np.sum(dx2, axis=0) # M x 1

    gradW2 = dW2
    gradb2 = db2
    gradW1 = dW1
    gradb1 = db1

    # print("========= shapes :  =============")
    # print("gradW2 gradb2 gradW1 gradb1")
    # print(gradW2.shape)
    # print(gradb2.shape)
    # print(gradW1.shape)
    # print(gradb1.shape)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + ( dimensions[1] + 1) * dimensions[2], )
    cost, grad = forward_backward_prop(data, labels, params, dimensions)
    print("cost")
    print(cost)
    # print("len(cost)")
    # print(len(cost))
    print("grad")
    print(grad)
    print("len(grad)")
    print(len(grad))
    print("param.shape")
    print(params.shape)
    ### YOUR CODE HERE
    # raise NotImplementedError

    ### END YOUR CODE

if __name__ == "__main__":
    # your_sanity_checks()
    sanity_check()
