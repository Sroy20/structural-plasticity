import numpy as np
import time
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels
img_size = X_train.shape[1]
num_class = 10

def initialze_parameters(img_size, num_class):
    theta = np.zeros([img_size, num_class])
    for i in range(num_class):
        temp_vals = theta[:,i]
        temp_indices = random.sample(range(0, img_size), 300)
        temp_vals[temp_indices[0:len(temp_indices)]] = 1
        #temp_vals[temp_indices[len(temp_indices)//2:len(temp_indices)]] = -1
        theta[:, i] = temp_vals

    return theta


def h_vec(theta, X):
    eta = np.matmul(X, theta)
    temp = np.exp(eta - np.reshape(np.amax(eta, axis=1), [-1, 1]))
    return (temp / np.reshape(np.sum(temp, axis=1), [-1, 1]))


# full vectorized
def GD_vec(theta, X_train, y_train, alpha):

    n_tar = 100
    del_w = -np.matmul(np.transpose(X_train), (h_vec(theta, X_train) - y_train))

    for i in range(num_class):
        this_integer_weights = theta[:, i]
        this_fitness = del_w[:, i]

        connected = np.nonzero(this_integer_weights)[0]
        random.shuffle(connected)
        connected = connected[0:n_tar]
        fitness_of_connected = this_fitness[connected]
        min_index_of_connected = np.argmin(fitness_of_connected)
        theta[connected[min_index_of_connected], i] -= 1

        not_connected = np.where(this_integer_weights == 0)[0]
        random.shuffle(not_connected)
        not_connected = not_connected[0:n_tar]
        fitness_of_not_connected = this_fitness[not_connected]
        max_index_of_not_connected = np.argmax(fitness_of_not_connected)
        theta[not_connected[max_index_of_not_connected], i] += 1

    return theta


def train_vec(X_train, y_train, max_iter, alpha):
    theta = initialze_parameters(img_size, num_class)
    for _ in range(max_iter):
        theta = GD_vec(theta, X_train, y_train, alpha)
    return theta


max_iter = 1000
alpha = 0.001
start = time.time()
theta = train_vec(X_train, y_train, max_iter, alpha)
end = time.time()
print("max iter", max_iter)
print("time elapsed: {0} seconds".format(end - start))
pred = np.argmax(h_vec(theta, X_test), axis=1)
print("percentage correct: {0}".format(np.sum(pred == np.argmax(y_test, axis=1)) / float(len(y_test))))

