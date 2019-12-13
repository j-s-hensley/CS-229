import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import progressbar


def loss(predictions, Y):
    """ Return the MSE loss. """
    return 0.5 * np.linalg.norm(Y - predictions)**2 /len(Y)

def avg_dif(predictions, Y):
    """ Calculate the average L1 distance between predictions and true ratings. """
    return sum([abs(Y[i] - predictions[i]) for i in range(len(Y))])[0] / len(Y)

def lin_reg_prediction(X, theta):
    """ Return the prediction vector for inputs X using theta output from linear regression. """
    return np.dot(X, theta)

def kernel_prediction(beta, K):
    """ Return the prediction vector for input kernel K using beta output from kernel methods. """
    return np.dot(beta.T, K).T

def calc_grad(X,Y,theta):
    """ Calculate the gradient for use in gradient descent for linear regression. """
    return np.dot((Y - np.dot(X,theta)).T,X).T / (X.shape[0])

def linear_regression(data, embedding, iters, learning_rate = 0.1, makePlot = True):
    """ Perform linear regression. """
    (X, X_val, X_test, Y, Y_val, Y_test) = data
    theta = np.zeros((X.shape[1],1))

    tloss = []
    vloss = []

    for i in progressbar.progressbar(iters):
        theta += learning_rate * calc_grad(X, Y, theta)

        tloss += [loss(lin_reg_prediction(X, theta), Y)]
        vloss += [loss(lin_reg_prediction(X_val, theta), Y_val)]
    if makePlot:
        plot(iters, tloss, vloss, embedding, "Linear Regression")
    test_predictions = lin_reg_prediction(X_test, theta)
    print("Average difference in star rating: %.4f" %avg_dif(test_predictions, Y_test))
    return tloss, vloss, loss(test_predictions, Y_test)

def Linear(xi, xj):
    """ Linear kernel function. """
    return np.dot(xi,xj.T)

def Square(xi, xj):
    """ Square kernel function. """
    return np.dot(xi,xj.T)**2

def Gaussian(xi, xj, sigma = 1):
    """ Gaussian kernel function. """
    return np.exp( - np.dot((xi-xj), (xi-xj).T) / (2 * sigma**2))

def kernel_method(data, kernel, embedding,  iters, learning_rate = 0.0001, makePlot = True):
    """ Use kernel method to learn parameter vector beta for prediction. """
    (K_train, K_val, K_test, Y_train, Y_val, Y_test) = data
    beta = np.zeros((K_train.shape[0],1))

    tloss = []
    vloss = []

    for i in progressbar.progressbar(iters):
        beta += learning_rate * ( Y_train - np.dot(beta.T, K_train).T )
        tloss += [loss(kernel_prediction(beta, K_train), Y_train)]
        vloss += [loss(kernel_prediction(beta, K_val), Y_val)]

    if makePlot:
        plot(iters, tloss, vloss, embedding, kernel.__name__+" Kernel")
    test_predictions = kernel_prediction(beta, K_test)
    print("Average difference in star rating: %.4f" %avg_dif(test_predictions, Y_test))
    return tloss, vloss, loss(test_predictions, Y_test)

def plot(iters, tloss, vloss, embedding, method):
    """ Make a plot of training and validation loss during training. """
    plt.plot(iters,tloss,label='training loss')
    plt.plot(iters,vloss,label='validation loss')

    plt.title('Average Loss per Example During Training for \n'+embedding+' Embedding with '+method)
    plt.xlabel('iteration number')
    plt.ylabel('average loss')
    plt.legend()
    plt.show()

def learning_rate_sweep(vals_to_test, func):
    """ Meta-function to sweep over possible learning rates and plot validation set loss. """
    outputs = []
    for i in vals_to_test:
        _, vloss, _ = func(i)
        outputs += [vloss[-1]]
    plt.plot(vals_to_test, outputs, 'o--')
    plt.title('Average Loss over Validation Set')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.show()
    print(outputs)

def load_data(embedding, trim = True):
    """ Load features as saved by features_X.py. """
    Y_train = np.expand_dims(np.load('Y_train.npy'), axis=1)
    Y_val = np.expand_dims(np.load('Y_val.npy'), axis=1)
    Y_test = np.expand_dims(np.load('Y_test.npy'), axis=1)

    if embedding == "One-Hot":
        X_train = sparse.load_npz('X_train_one_hot.npz').todense()
        X_val = sparse.load_npz('X_val_one_hot.npz').todense()
        X_test = sparse.load_npz('X_test_one_hot.npz').todense()

    if embedding == "BERT":
        X_train = np.load('X_train_bert.npy')
        X_val = np.load('X_val_bert.npy')
        X_test = np.load('X_test_bert.npy')

    if trim:
        X_train = X_train[:5000,:]
        X_val = X_val[:1000,:]
        X_test = X_test[:1000,:]

        Y_train = Y_train[:5000,:]
        Y_val = Y_val[:1000,:]
        Y_test = Y_test[:1000,:]

    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

def load_kernel(embedding, kernel):
    """ Load kernel matrix as saved by features_X.py. """
    Y_train = np.expand_dims(np.load('Y_train.npy'), axis=1)[:5000,:]
    Y_val = np.expand_dims(np.load('Y_val.npy'), axis=1)[:1000,:]
    Y_test = np.expand_dims(np.load('Y_test.npy'), axis=1)[:1000,:]

    K_train = np.load('K_train_'+embedding+'_'+kernel.__name__+'.npy')
    K_val = np.load('K_val_'+embedding+'_'+kernel.__name__+'.npy')
    K_test = np.load('K_test_'+embedding+'_'+kernel.__name__+'.npy')

    return (K_train, K_val, K_test, Y_train, Y_val, Y_test)

def pretty_print(inps):
    """ Print out training, validation, and test set losses. """
    train_loss, validation_loss, test_loss = inps
    print("Training loss: %.4f   Validation loss: %.4f   Test loss: %.4f" %(train_loss[-1], validation_loss[-1], test_loss))


if __name__ == "__main__":
    pretty_print(linear_regression(load_data("One-Hot"), "One-Hot", range(50), 0.5))

    pretty_print(kernel_method(load_kernel("One-Hot", Linear), Linear, "One-Hot", range(50), 0.0002))

    pretty_print(kernel_method(load_kernel("One-Hot", Gaussian), Gaussian, "One-Hot", range(100), 0.015))

    pretty_print(linear_regression(load_data("BERT"), "BERT", range(100), 0.02))

    pretty_print(kernel_method(load_kernel("BERT", Linear), Linear, "BERT", range(100), 4*10**-6))

    pretty_print(kernel_method(load_kernel("BERT", Gaussian), Gaussian, "BERT", range(100), 0.05))

    pretty_print(kernel_method(load_kernel("One-Hot", Square), Square, "One-Hot", range(100), 0.5*10**-4))

    pretty_print(kernel_method(load_kernel("BERT", Square), Square, "BERT", range(100), 3*10**-8))


    #vals = [10**-8, 5*10**-8, 10**-7, 5*10**-7, 10**-6, 5*10**-6, 10**-5, 5*10**-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    #lr_func = lambda lr: linear_regression(load_data("BERT"), "BERT", range(100), lr, makePlot=False)
    #learning_rate_sweep(vals, lr_func)

    #k_func = lambda lr: kernel_method(load_kernel("BERT", Square), Square, "BERT", range(100), lr, makePlot=True)
    #learning_rate_sweep(vals, k_func)
