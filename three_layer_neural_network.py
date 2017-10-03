__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        # print(type)
        # print(z)
        out = 0
        if type == 'tanh':
            out = np.tanh(z)
            # if np.isnan(out):
            # print("self.b2")
            # print(self.b2)
            # print("out")
            # print(out)
            # print("np.exp(-z)")
            # print(np.exp(-z))
            # print("z")
            # print(z)
            # print("self.probs")
            # print(self.probs)
            # input()
        elif type == 'Sigmoid':
            out = 1 / (1 + np.exp(-z))

        elif type == 'ReLU':
            # print(z)
            out = z * (z > 0)  # max([0,z])# #
            # if z >= 0
            #    out = z
            # else
            #    out = 0
        else:
            # print(type)
            print("Your type is not correct. It is " + type)
        # print("out")
        # print(out)
        return out

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        out = 0
        if type == 'tanh':
            out = (1 / np.cosh(z)) ** 2
        elif type == 'Sigmoid':
            out = np.exp(z) / ((1 + np.exp(z)) ** 2)  # (1+np.exp(z))) #(2/(np.exp(z)+np.exp(-z)))
        elif type == 'ReLU':
            out = z
            # print(out[0][0])
            # delta3[range(num_examples), y]
            for outrow in range(len(out)):
                for outcol in range(len(out[0])):
                    if out[outrow][outcol] > 0:
                        out[outrow][outcol] = 1
                    else:
                        out[outrow][outcol] = 0
                        # out = np.where(z>0)
                        # print(out)
                        # input()
                        # if z >= 0:
                        #    out = 1
                        # else:
                        #    out = 0
        else:
            # print(type)
            print("Your type is not correct. It is " + type)
        return out

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        # print(actFun)
        # YOU IMPLEMENT YOUR feedforward HERE
        self.z1 = np.dot(X, self.W1) + self.b1  # np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        # self.a2 = actFun(self.z2)

        # print("hahaha " + actFun)
        # self.z1 = np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        # print(X)
        # self.a1 = self.z1
        # self.z2 = np.zeros((len(X),self.nn_output_dim))
        # print(self.z1)
        # loop through each example
        # for k in range(len(X)):
        # loop through each dimension of each sample
        #    for i in range(self.nn_hidden_dim):
        # loop through inputs for each sample
        #        for n in range(len(X[k])):
        #            self.z1[k][i] += self.W1[n][i]*X[k][n]
        #        self.z1[k][i] += self.b1[0][i]
        # print("z1")
        # print(self.z1[k][i])
        # print("actFun")
        # print(actFun(self.z1[k][i]))
        #        self.a1[k][i] = actFun(self.z1[k][i])
        # print(self.b2)
        #    for j in range(len(self.W2[0])):
        #        for b in range(self.nn_hidden_dim):
        #            self.z2[k][j] += self.W2[b][j]*self.a1[k][b]
        #        self.z2[k][j]+=self.b2[0][j]

        # self.a2 = softmax(self.z2)
        exp_scores = np.exp(self.z2)
        # print(np.sum(exp_scores, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # for i in range(len(exp_scores)):

        # print(self.probs)
        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        # num_examples = len(X)
        data_loss = 0
        for n in range(num_examples):
            for i in range(self.nn_output_dim):
                # print("Weeelll!!")
                data_loss += y[n] * np.log(self.probs[n][i])  # np.linalg.norm(self.probs-y)
                # print("a1")
                # print(self.a1)
                # print("b1")
                # print(self.b1)
                # print("b2")
                # print(self.b2)
                # print("exp_scores")
                # print(exp_scores)
                # print("exp_scores cell")
                # print(exp_scores[n][k])

                if np.isnan(data_loss):
                    print("np.log")
                    print(np.log(self.probs[n][i]))
                    print("y[k]")
                    print(y[n])
                    input()

        data_loss = -data_loss / num_examples
        # Add regulatization term to loss (optional)

        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        delta3 = self.probs
        # print(delta3)
        delta3[range(num_examples), y] -= 1
        delta3 /= num_examples

        # a1Trans = np.transpose(self.a1)
        # a1Transnp.transpose()
        dW2 = np.dot(np.transpose(self.a1), delta3)
        # print(dW2)
        db2 = np.sum(delta3, axis=0)
        # print(self.z1)
        # input()
        # delta2 = delta3.dot(self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        delta2 = np.dot(delta3, np.transpose(self.W2)) * self.diff_actFun(self.z1, self.actFun_type)
        # print(delta2)
        # input()
        dW1 = np.dot(np.transpose(X), delta2)
        db1 = np.sum(delta2, axis=0)

        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.00005, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        # print("welll " + self.actFun_type)
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)
            # print("y")
            # print(y[1])
            # print(len(y))
            # input()
            # Add regularization terms (b1 and b2 don't have regularization terms)
            # print(self.reg_lambda)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2
            # for q in range(len(self.W1[0])):
            # if not db2.all() or not db1.all() or not dW2.all() or not dW1.all():
            #    print("db2")
            #    print(db2)
            #    print("db1")
            #    print(db1)
            #    print("dW2")
            #    print(dW2)
            #    print("dW1")
            #    print(dW1)
            # print("self.probs")
            # print(self.probs)
            #    input()
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
    # network1 = NeuralNetwork(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    # print(y)

    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()