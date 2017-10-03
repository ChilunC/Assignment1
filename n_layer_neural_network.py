__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork


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

class Layer(NeuralNetwork):
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
        # np.random.seed(seed)
        # self.W1 = W1#np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        # self.b1 = b1 #np.zeros((1, self.nn_hidden_dim))
        # self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        # self.b2 = np.zeros((1, self.nn_output_dim))

    def feedforward(self, X, W1, b1, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        # print(actFun)
        # YOU IMPLEMENT YOUR feedforward HERE
        # print("hahaha " + actFun)
        # print(b1)
        self.z1 = np.dot([X], W1) + b1  # np.zeros((1,self.nn_hidden_dim)) #len(self.W1[0])))
        # print(X)
        self.a1 = self.actFun(self.z1, self.actFun_type)

        return self.z1, self.a1

    def backprop(self, X, W1, z1, deltan):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        # num_examples = len(X)
        # delta3 = self.probs
        # delta3[range(num_examples), y] -= 1
        # expsum = 0
        # expmultsum = 0

        # exp_scores = np.exp(self.z2)

        deltan1 = np.dot(deltan, np.transpose(W1)) * self.diff_actFun(z1, self.actFun_type)
        return deltan1


class DeepNeuralNetwork(NeuralNetwork):
    # class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, nn_layers_dim, actFun_type='tanh', reg_lambda=0.01,
                 seed=0):
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
        self.nn_layers_dim = nn_layers_dim - 1
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.Layer1 = Layer(self.nn_hidden_dim, self.nn_hidden_dim, self.nn_hidden_dim, self.actFun_type)

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        if self.nn_layers_dim:
            self.Wn = np.random.randn(self.nn_layers_dim, self.nn_hidden_dim, self.nn_hidden_dim) / np.sqrt(
                self.nn_hidden_dim)
            self.bn = np.zeros((self.nn_layers_dim, self.nn_hidden_dim))
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

        if type == 'tanh':
            out = 1 / (1 + np.exp(-z))

        elif type == 'Sigmoid':
            out = np.tanh(z)
        elif type == 'ReLU':
            out = z

            for n in range(self.nn_layers_dim):
                # print(z[n])
                out[n] = z[n] * (z[n] > 0)  # max([0,z]) # max([0,z[n]])
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
        # print("hahaha " + actFun)

        if self.nn_layers_dim:
            self.zn = np.zeros((self.nn_layers_dim, len(X), self.nn_hidden_dim))
            self.an = np.zeros((self.nn_layers_dim, len(X), self.nn_hidden_dim))

        self.z1 = np.dot(X, self.W1) + self.b1  # np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        self.a1 = actFun(self.z1)
        if self.nn_layers_dim:
            self.zn[0], self.an[0] = self.Layer1.feedforward(self.a1, self.Wn[0], self.bn[0],
                                                             lambda x: self.actFun(x, type=self.actFun_type))

            if self.nn_layers_dim > 1:
                for d in range(self.nn_layers_dim - 1):
                    self.zn[d + 1], self.an[d + 1] = self.Layer1.feedforward(self.an[d], self.Wn[d + 1], self.bn[d + 1],
                                                                             lambda x: self.actFun(x,
                                                                                                   type=self.actFun_type))
        self.z2 = np.dot(self.an[self.nn_layers_dim - 1],
                         self.W2) + self.b2  # np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        self.a2 = actFun(self.z2)

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

        data_loss = -data_loss / num_examples
        # Add regulatization term to loss (optional)

        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1[0])) + np.sum(np.square(self.W2)))
        if self.nn_layers_dim:
            for l in range(self.nn_layers_dim):
                data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.Wn[l])))
        return (1. / num_examples) * data_loss

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE


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

        delta2 = np.dot(delta3, np.transpose(self.W2)) * self.diff_actFun(self.zn[self.nn_layers_dim - 1],
                                                                          self.actFun_type)
        dbn = np.zeros((self.nn_layers_dim, self.nn_hidden_dim))
        dWn = np.zeros(
            (self.nn_layers_dim, self.nn_hidden_dim, self.nn_hidden_dim))  # np.zeros((1, self.nn_layers_dim))
        for n in range(self.nn_layers_dim):
            dWn[self.nn_layers_dim - 1 - n] = np.dot(np.transpose(self.an[self.nn_layers_dim - 1 - n]), delta2)
            dbn[self.nn_layers_dim - 1 - n] = np.sum(delta2, axis=0)
            delta2 = self.Layer1.backprop(self.an[self.nn_layers_dim - 1 - n], self.Wn[self.nn_layers_dim - 1 - n],
                                          self.zn[self.nn_layers_dim - 1 - n], delta2)
        # print(delta2)
        # input()
        dW1 = np.dot(np.transpose(X), delta2)
        db1 = np.sum(delta2, axis=0)

        ########################################################


        return dW1, dW2, db1, db2, dWn, dbn

    def fit_model(self, X, y, epsilon=0.0001, num_passes=20000, print_loss=True):
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
            dW1, dW2, db1, db2, dWn, dbn = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            # print(self.reg_lambda)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1
            dWn += self.reg_lambda * self.Wn

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2
            self.Wn += -epsilon * dWn
            self.bn += -epsilon * dbn

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
                # print("self.b2")
                # print(self.b2)
                # print("self.W2")
                # print(self.W2)
                # print("self.W1[n][i]")
                # print(self.W1)
                # print("self.b1")
                # print(self.b1)
                # print("self.Wn[n][i]")
                # print(self.Wn)
                # print("self.b1")
                # print(self.bn)


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
    # network1 = NeuralNetwork(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    # print(y)

    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, nn_layers_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, nn_layers_dim=5, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()