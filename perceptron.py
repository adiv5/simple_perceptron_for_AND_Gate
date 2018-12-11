# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:42:25 2018

@author: Adi
"""

import math
import scipy.special
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])
bias = 1
weights = np.ones(2)
rate = 0.1
iteration = 10000
for index in range(iteration):
    print("Iteration {}".format(index))
    for i in range(len(inputs)):
        print("Input {0}, Weights {1}, Bias {2}".format(inputs[i], weights, bias), end=' ')
        net = np.inner(inputs[i], weights) + bias
        output = scipy.special.expit(net)
        print("Net {0}, Output {1}, Target {2}".format(net, output, targets[i]))
        error = rate*(targets[i]-output)
        weights = np.add(weights, (error*inputs[i]))
        bias = bias + error