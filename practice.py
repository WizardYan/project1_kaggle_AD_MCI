import numpy as np
import matplotlib.pyplot as plt
import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
print X.shape
print Y.shape