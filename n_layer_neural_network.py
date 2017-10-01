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
        np.random.seed(seed)
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
        self.z1 = np.zeros((1, self.nn_hidden_dim))  # len(self.W1[0])))
        # print(X)
        self.a1 = self.z1
        # self.z2 = np.zeros((len(X),self.nn_output_dim))
        # print(self.z1)
        # loop through each example
        # for k in range(len(X)):
        # loop through each dimension of each sample
        for i in range(len(W1[0])):
            # loop through inputs for each sample
            # print(W1)
            # print(X)
            # print()
            for n in range(len(W1)):  # len(X)):
                self.z1[0][i] += W1[n][i] * X[n]
            self.z1[0][i] += b1[0][i]
            # print("z1")
            # print(self.z1[k][i])
            # print("actFun")
            # print(actFun(self.z1[k][i]))
            self.a1[0][i] = actFun(self.z1[0][i])
            # print(self.b2)
            # for j in range(len(self.W2[0])):
            #    for b in range(self.nn_hidden_dim):
            #        self.z2[k][j] = self.W2[b][j]*self.a1[k][b]
            #    self.z2[k][j]+=self.b2[0][j]

        # self.a2 = softmax(self.z2)
        # exp_scores = np.exp(self.z2)
        # print(np.sum(exp_scores, axis=1, keepdims=True))
        # self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # for i in range(len(exp_scores)):

        # print(self.probs)
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
        # print(np.sum(exp_scores, axis=1, keepdims=True))
        # expsum = np.sum(exp_scores, axis=1, keepdims=True)
        # expmultsum = 0

        # for n in range(num_examples):
        # expsum += np.exp(self.z[n])
        #    expmultsum += np.exp(self.z2[n])*self.z2[n]

        # dW2test = np.zeros((len(X),len(self.W2),len(self.W2[0])))
        # db2test = np.zeros((len(X),len(self.b2),len(self.b2[0])))

        # leave in case want to return
        # dW1test = np.zeros((len(self.W1),len(self.W1[0])))
        deltan1 = np.dot(deltan, np.transpose(W1)) * self.diff_actFun(z1, self.actFun_type)
        return deltan1

        # dW1test = np.zeros((len(W1),len(W1[0])))
        # dWntest = np.zeros((len(W1),len(W1[0])))
        # dbntest = np.zeros((1,len(W1[0])))

        # db1test = np.zeros((len(X),len(self.b1),len(self.b1[0])))
        # print(z1)
        # for w in range(len(X)):
        #    for e in range(len(W1[0])):
        #            dbntest[0][w] += W1[e][w]*self.diff_actFun(z1[w], self.actFun_type)*hisderiv[w][e]
        #    for u in range(len(W1[0])):
        #        dbntest[0][u] = self.diff_actFun(z1[u], self.actFun_type)

        #        dWntest[w][u] = X[w]*dbntest[0][w]
        # do the same as finding the current state, but derive with respect to the z and keep the dimensions ie don't sum
        #        dW1test[w][u] = W1[u][w]*self.diff_actFun(z1[w], self.actFun_type)*hisderiv[w][u]#dW1test[w][u]*W1[w][u]
        # for p in range(len(X)):
        # dWntest[w][u] = dbntest[0][u]*X[w] #dW1test[w][u]*X[w]


        # dW1test is the derivative of the next layer, dWntest is the derivative of the current layer, dbntest is the derivative of the current layer b
        # return dW1test, dWntest, dbntest


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
        self.b1 = np.zeros((self.nn_layers_dim, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

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
        self.z1 = np.zeros((len(X), self.nn_hidden_dim))  # len(self.W1[0])))
        if self.nn_layers_dim:
            self.zn = np.zeros((len(X), self.nn_layers_dim, self.nn_hidden_dim))
            self.an = np.zeros((len(X), self.nn_layers_dim, self.nn_hidden_dim))
        # print(self.nn_layers_dim)
        # print(self.an[0])
        # print(X)
        self.a1 = self.z1
        self.z2 = np.zeros((len(X), self.nn_output_dim))
        # print(self.z1)
        # loop through each example
        for k in range(len(X)):
            # loop through each dimension of each sample
            for i in range(self.nn_hidden_dim):
                # loop through inputs for each sample
                for n in range(len(X[k])):
                    self.z1[k][i] += self.W1[n][i] * X[k][n]
                self.z1[k][i] += self.b1[0][i]

                self.a1[k][i] = actFun(self.z1[k][i])
            # if more than one layer get the feedforward values
            if self.nn_layers_dim:
                # print(self.a1[k])
                self.zn[k][0], self.an[k][0] = self.Layer1.feedforward(self.a1[k], self.W1, self.b1, actFun)
                # for i in range(self.nn_hidden_dim):
                # self.zn[k][0][i] += self.W1[1][n][i]*self.a1[k][i]
                # self.an[k][0][i] = actFun(self.zn[k][0][i])
                # if more than two layers loop until all layers are found.
                # print(self.an[k])
                if self.nn_layers_dim > 1:
                    for d in range(self.nn_layers_dim - 1):
                        self.zn[k][d + 1], self.an[k][d + 1] = self.Layer1.feedforward(self.an[k][d], self.Wn[d],
                                                                                       self.bn[d], actFun)
                        # self.zn[k][d+1][i] += self.W1[2+d][n][i]*self.zn[k][d][i]
                        # self.an[k][d+1][i] = actFun(self.zn[k][d+1][i])
                # print(self.an[k])
                # for i in range(self.nn_hidden_dim):
                #    self.zn[k][0][i] += self.b1[1][i]
                #    if self.nn_layers_dim>2:
                #        for d in range(self.nn_layers_dim-2):
                #            self.zn[k][d+1][i] += self.b1[2+d][i]
                # print("z1")
                # print(self.z1[k][i])
                # print("actFun")
                # print(actFun(self.z1[k][i]))

                # print(self.an[k])
                # loop through output and hidden dim or final layer weights to get the output
                for j in range(len(self.W2[0])):
                    for b in range(self.nn_hidden_dim):
                        self.z2[k][j] += self.W2[b][j] * self.an[k][self.nn_layers_dim - 1][b]
                    self.z2[k][j] += self.b2[0][j]
            else:
                # if no hidden layer, just jump to calculating the output
                for j in range(len(self.W2[0])):
                    for b in range(self.nn_hidden_dim):
                        self.z2[k][j] += self.W2[b][j] * self.a1[k][b]
                    self.z2[k][j] += self.b2[0][j]

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
        # print(delta3)
        # delta2 = delta3.dot(self.W2.T) * self.diff_actFun(self.z1, type=self.actFun_type)
        delta2 = np.dot(delta3, np.transpose(self.W2)) * self.diff_actFun(self.zn[self.nn_layers_dim], self.actFun_type)
        dbn = np.zeros((self.nn_layers_dim, self.nn_hidden_dim))
        dWn = np.zeros(
            (self.nn_layers_dim, self.nn_hidden_dim, self.nn_hidden_dim))  # np.zeros((1, self.nn_layers_dim))
        for n in range(self.nn_layers_dim - 1):
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

    def fit_model(self, X, y, epsilon=0.00001, num_passes=20000, print_loss=True):
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
            for q in range(len(self.W1[0])):
                if np.isnan((self.W1[0][q])):
                    print("self.b2")
                    print(self.b2)
                    print("self.W2")
                    print(self.W2)
                    print("self.W1[n][i]")
                    print(self.W1)
                    print("self.b1")
                    print(self.b1)
                    # print("self.probs")
                    # print(self.probs)
                    input()
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    # self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
    # network1 = NeuralNetwork(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    # print(y)

    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, nn_layers_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, nn_layers_dim=2, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()