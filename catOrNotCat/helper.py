import numpy as np


# helper functions for functions involved in guessing the values of w and b to finally predict the values

# for a single training example we have y= 0 if wx+b<0.5 y=1 if wx+b>=0.5 if
# 1 image is 64*64*3 values(64*64 for r,g,b) then
# training set data x is a matrix of 12288 rows and 209(here) columns that is number of features * number of samples
# training set data y is vector of 1*number of samples
# test set data is a

# converting the sample data which is number_of_sample*64*64*3 to a matrix of size 64*64*3 * number of samples
def flatten_multi_dimesion_data(X):
    return X.reshape(X.shape[0], -1).T


def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def tanh(z):
    return np.divide(np.exp(z) - np.exp(-z), np.exp(z) + np.exp(-z))

def relu(z):
    return max(0,z)

def accuracy(predict, actual):
    error = np.mean(np.abs(predict - actual)) * 100
    return 100 - error


# steps of estimating value of y
# predict w and b  :  w dimesion should be number of samples * 1 and b can be a real number
# using w,b on the sigmoid function wx+b to obtain the value y
# if y>0.5 y=1 else 0


def initialise(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# x is the matrix of n samples x is effectively the entire 12288*n data
def gradient_descent(x, y, w, b):
    n = x.shape[1]  # n = number of samples in training set
    # estimating the value of y based on w and b
    A = sigmoid(np.dot(w.T, x) + b)

    ##note here np.log giving nan for some reason idk why?
    J = -np.mean(y*np.log(A)+(1-y)*np.log(1-A))
    dz = A - y
    w_gradient = 1 / n * np.dot(x, dz.T)
    b_gradient = np.mean(dz)
    return J, w_gradient, b_gradient


# once done initialising, figure out the value of w and b using the training data using optimization

def optimization(train_x, train_y, num_itr, learning_rate, w, b):
    # we only check with the gradient until either the num_itr crosses or a minimum for cost is hit
    for i in range(num_itr):

        cost, w_gradient, b_gradient = gradient_descent(train_x, train_y, w, b)

        # if i % 10 == 0:
        #     print("is decr" + str(cost))

        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient

    # finally i have the optimal value of w and b
    return w, b


def model(train_x_uf, train_y, test_x_uf, test_y):

    train_x=flatten_multi_dimesion_data(train_x_uf)
    test_x=flatten_multi_dimesion_data(test_x_uf)
    test_x=test_x[:,:test_y.shape[1]]
    #print(f"test x dim {test_x.shape}")
    dim = train_x.shape[0]
    w, b = initialise(dim)
    w, b = optimization(train_x, train_y, 200, 0.005, w, b)

    # now the prediction i make for train_x will be
    y_predict_train = sigmoid(np.dot(w.T, train_x) + b)
    y_predict_train = (y_predict_train >= 0.5) * 1.0
    print(accuracy(y_predict_train, train_y))

    y_predict_test = sigmoid(np.dot(w.T, test_x) + b)
    y_predict_test = (y_predict_test >= 0.5) * 1.0
    print(accuracy(y_predict_test, test_y))
