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
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
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
        #self.W1 = W1#np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        #self.b1 = b1 #np.zeros((1, self.nn_hidden_dim))
        #self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        #self.b2 = np.zeros((1, self.nn_output_dim))
        
        
    def feedforward(self, X, W1, b1, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        #print(actFun)
        # YOU IMPLEMENT YOUR feedforward HERE
        #print("hahaha " + actFun)
        self.z1 = np.zeros((1,self.nn_hidden_dim)) #len(self.W1[0])))
        #print(X)
        self.a1 = self.z1
        #self.z2 = np.zeros((len(X),self.nn_output_dim))
        #print(self.z1)
        #loop through each example
        #for k in range(len(X)):
            #loop through each dimension of each sample
        for i in range(len(W1[0])):
            #loop through inputs for each sample
            #print(W1)
            #print(X)
            #print()
            for n in range(len(W1)): #len(X)):            
                self.z1[0][i] += W1[n][i]*X[n]
            self.z1[0][i] += b1[0][i]
            #print("z1")
            #print(self.z1[k][i])
            #print("actFun")
            #print(actFun(self.z1[k][i]))
            self.a1[0][i] = actFun(self.z1[0][i])
            #print(self.b2)
            #for j in range(len(self.W2[0])):
            #    for b in range(self.nn_hidden_dim):
            #        self.z2[k][j] = self.W2[b][j]*self.a1[k][b]
            #    self.z2[k][j]+=self.b2[0][j]
                
        #self.a2 = softmax(self.z2)
        #exp_scores = np.exp(self.z2)
        #print(np.sum(exp_scores, axis=1, keepdims=True))
        #self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #for i in range(len(exp_scores)):
            
        #print(self.probs)
        return self.z1, self.a1
        

    def backprop(self, X, W1, z1, hisderiv):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        #num_examples = len(X)
        #delta3 = self.probs
        #delta3[range(num_examples), y] -= 1
        #expsum = 0
        #expmultsum = 0
        
        #exp_scores = np.exp(self.z2)
        #print(np.sum(exp_scores, axis=1, keepdims=True))
        #expsum = np.sum(exp_scores, axis=1, keepdims=True)
        #expmultsum = 0
        
        #for n in range(num_examples):
            #expsum += np.exp(self.z[n])
        #    expmultsum += np.exp(self.z2[n])*self.z2[n]
        
        #dW2test = np.zeros((len(X),len(self.W2),len(self.W2[0])))
        #db2test = np.zeros((len(X),len(self.b2),len(self.b2[0])))
        
        #leave in case want to return
        #dW1test = np.zeros((len(self.W1),len(self.W1[0])))
        dW1test = np.zeros((len(W1),len(W1[0])))
        dWntest = np.zeros((len(W1),len(W1[0])))
        dbntest = np.zeros((1,len(W1[0])))
        
        #db1test = np.zeros((len(X),len(self.b1),len(self.b1[0])))
        print(z1)
        for w in range(len(X)):
            for e in range(len(W1[0])):
                    dbntest[0][w] += W1[e][w]*self.diff_actFun(z1[w], self.actFun_type)*hisderiv[w][e]
            for u in range(len(W1[0])):
                dbntest[0][u] = self.diff_actFun(z1[u], self.actFun_type)
                
                dWntest[w][u] = X[w]*dbntest[0][w]
                #do the same as finding the current state, but derive with respect to the z and keep the dimensions ie don't sum
                dW1test[w][u] = W1[u]][w]*self.diff_actFun(z1[w], self.actFun_type)*hisderiv[w][u]#dW1test[w][u]*W1[w][u]
                #for p in range(len(X)):
                #dWntest[w][u] = dbntest[0][u]*X[w] #dW1test[w][u]*X[w]
                
        
        #dW1test is the derivative of the next layer, dWntest is the derivative of the current layer, dbntest is the derivative of the current layer b
        return dW1test, dWntest, dbntest 
        
        



class DeepNeuralNetwork(NeuralNetwork):
#class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, nn_layers_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
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
        self.Layer1 = Layer(self.nn_hidden_dim, self.nn_hidden_dim , self.nn_hidden_dim, self.actFun_type)
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        if self.nn_layers_dim:
            self.Wn = np.random.randn(self.nn_layers_dim, self.nn_hidden_dim, self.nn_hidden_dim) / np.sqrt(self.nn_hidden_dim)
            self.bn = np.zeros((1, self.nn_layers_dim))
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
        #print(actFun)
        # YOU IMPLEMENT YOUR feedforward HERE
        #print("hahaha " + actFun)
        self.z1 = np.zeros((len(X),self.nn_hidden_dim)) #len(self.W1[0])))
        if self.nn_layers_dim:
            self.zn = np.zeros((len(X),self.nn_layers_dim,self.nn_hidden_dim))
            self.an = np.zeros((len(X),self.nn_layers_dim,self.nn_hidden_dim))
        #print(self.nn_layers_dim)
        #print(self.an[0])
        #print(X)
        self.a1 = self.z1
        self.z2 = np.zeros((len(X),self.nn_output_dim))
        #print(self.z1)
        #loop through each example
        for k in range(len(X)):
            #loop through each dimension of each sample
            for i in range(self.nn_hidden_dim):
                #loop through inputs for each sample
                for n in range(len(X[k])):            
                    self.z1[k][i] += self.W1[n][i]*X[k][n]
                self.z1[k][i] += self.b1[0][i]
                
                self.a1[k][i] = actFun(self.z1[k][i])
            #if more than one layer get the feedforward values    
            if self.nn_layers_dim:
                #print(self.a1[k])
                self.zn[k][0], self.an[k][0] = self.Layer1.feedforward(self.a1[k], self.W1, self.b1, actFun)
                #for i in range(self.nn_hidden_dim):
                    #self.zn[k][0][i] += self.W1[1][n][i]*self.a1[k][i]
                    #self.an[k][0][i] = actFun(self.zn[k][0][i])
                #if more than two layers loop until all layers are found.
                #print(self.an[k])
                if self.nn_layers_dim>1:
                    for d in range(self.nn_layers_dim-1):
                        self.zn[k][d+1], self.an[k][d+1] = self.Layer1.feedforward(self.an[k][d], self.Wn[d], self.bn[d], actFun)
                        #self.zn[k][d+1][i] += self.W1[2+d][n][i]*self.zn[k][d][i]
                    #self.an[k][d+1][i] = actFun(self.zn[k][d+1][i])
                #print(self.an[k])        
                #for i in range(self.nn_hidden_dim):   
                #    self.zn[k][0][i] += self.b1[1][i]
                #    if self.nn_layers_dim>2:
                #        for d in range(self.nn_layers_dim-2):                      
                #            self.zn[k][d+1][i] += self.b1[2+d][i]
                #print("z1")
                #print(self.z1[k][i])
                #print("actFun")
                #print(actFun(self.z1[k][i]))
                
                #print(self.an[k])
                #loop through output and hidden dim or final layer weights to get the output
                for j in range(len(self.W2[0])):
                    for b in range(self.nn_hidden_dim):
                        self.z2[k][j] += self.W2[b][j]*self.an[k][self.nn_layers_dim-1][b]
                    self.z2[k][j]+=self.b2[0][j]
            else:
                #if no hidden layer, just jump to calculating the output
                for j in range(len(self.W2[0])):
                    for b in range(self.nn_hidden_dim):
                        self.z2[k][j] += self.W2[b][j]*self.a1[k][b]
                    self.z2[k][j]+=self.b2[0][j]
                
        #self.a2 = softmax(self.z2)
        exp_scores = np.exp(self.z2)
        #print(np.sum(exp_scores, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #for i in range(len(exp_scores)):
            
        #print(self.probs)
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
        #num_examples = len(X)
        data_loss=0
        for n in range(num_examples):
            for i in range(self.nn_output_dim):
                #print("Weeelll!!")
                data_loss += y[n]*np.log(self.probs[n][i]) #np.linalg.norm(self.probs-y)
                #print("a1")
                #print(self.a1)
                #print("b1")
                #print(self.b1)
                #print("b2")
                #print(self.b2)
                #print("exp_scores")
                #print(exp_scores)
                #print("exp_scores cell")
                #print(exp_scores[n][k])
                
                if np.isnan(data_loss):
                    print("np.log")
                    print(np.log(self.probs[n][i]))
                    print("y[k]")
                    print(y[n])
                    input()
                    
        data_loss = -data_loss/num_examples
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
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        expsum = 0
        expmultsum = 0
        
        exp_scores = np.exp(self.z2)
        #print(np.sum(exp_scores, axis=1, keepdims=True))
        expsum = np.sum(exp_scores, axis=1, keepdims=True)
        #expmultsum = 0
        
        #for n in range(num_examples):
            #expsum += np.exp(self.z[n])
        #    expmultsum += np.exp(self.z2[n])*self.z2[n]
        
        dW2test = np.zeros((len(X),len(self.W2),len(self.W2[0])))
        db2test = np.zeros((len(X),len(self.b2),len(self.b2[0])))
        dW1test = np.zeros((len(X),len(self.W1),len(self.W1[0])))
        db1test = np.zeros((len(X),len(self.b1),len(self.b1[0])))
        dWntest = np.zeros((len(X),self.nn_layers_dim, len(self.W1),len(self.W1[0])))
        dbntest = np.zeros((len(X),self.nn_layers_dim, len(self.b1),len(self.b1[0])))
        
        dW2 = 0
        db2 = 0
        dW1 = 0
        db1 = 0
        
        #loop through all examples
        for n in range(num_examples):
            #create derivative variable for w2 matrix
            expmultsumdz2w2 = np.zeros((len(self.W2),len(self.W2[0])))
            for w in range(len(self.W2)):
                for t in range(len(self.W2[0])):
                    expmultsumdz2w2[w][t] = self.a1[n][w]*np.exp(self.z2[n][t])
            
            #create derivative variable for b2 matrix
            expmultsumdz2b2 = np.zeros(len(self.b2[0]))
            for w in range(len(self.b2[0])):
                #for t in range(self.nn_output_dim):
                expmultsumdz2b2[w] = np.exp(self.z2[n][w])
                
            if self.nn_layers_dim:  
                #create the multiplier factor for the derivative of the loss function
                expmultsumdz2wn = np.ones((self.nn_layers_dim, len(self.Wn[0]),len(self.Wn[0][0])))
                #expmultsumdz2w1
                #multiz2w #np.zeros((len(self.W1),len(self.W1[0])))
                mulz2fac = 0
                #why are we iterating over the hidden dimension twice?
                #because the Wn is twice the hidden dimension rather than input and hidden like W1
                #loop through getting the output multiplication factor
                for w in range(self.nn_hidden_dim):
                    for u in range(len(self.Wn[0][0])):
                        for t in range(self.nn_output_dim):
                            #create the exponential times the output weight factor
                            mulz2fac += np.exp(self.z2[n][t])*self.W2[u][t]
                        #stores values into a matrix to use for calculations
                        expmultsumdz2wn[self.nn_layers_dim-1][w][u] = expmultsumdz2wn[self.nn_layers_dim-1][w][u]*mulz2fac
                        
                        
                #create multiplication factor that multiplies by the previous layer derivative to multiply by the current layer derivative
                dWnNextDtemp = expmultsumdz2wn[self.nn_layers_dim-1] #np.ones((len(self.Wn[0]),len(self.Wn[0][0])))
                dbnNextDtemp = expmultsumdz2wn[self.nn_layers_dim-1][0] #np.zeros((1,len(self.b1[0])))
                expmultsumdz2bn = np.zeros((self.nn_layers_dim,len(self.b1[0])))
                for v in range(self.nn_layers_dim-1):
                    
                    #dW1test is the derivative of the next layer, dWntest is the derivative of the current layer, dbntest is the derivative of the current layer b
                    #return dW1test, dWntest, dbntest
                    dWnNextDeriv, dWnCurDeriv, dbnCurDeriv = self.Layer1.backprop(self.an[self.nn_layers_dim-v],self.Wn[self.nn_layers_dim-v],self.b1[self.nn_layers_dim-v-1],self.zn[n][self.nn_layers_dim-v-1], dWnNextDtemp) #backprop(self, X, W1, z1, hisderiv)
                    #loop through to create the bn multiplier
                    for w in range(self.nn_hidden_dim):
                        expmultsumdz2bn[self.nn_layers_dim-v-1][w] = dWnNextDeriv[w][u]*dbnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                        #loop through to create the wn multiplier
                        for u in range(len(self.Wn[0][0])):                        
                            expmultsumdz2wn[self.nn_layers_dim-v][w][u] = dWnNextDeriv[w][u]*dWnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                            
                            dWnNextDtemp[w][u] = dWnNextDtemp[w][u]*dWnNextDeriv[w][u]
                            #expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type) #expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)
                            
                #create derivative variable for w1 matrix
                expmultsumdz2w1 = np.zeros((len(self.W1),len(self.W1[0])))
                expmultsumdz2b1 = np.ones((1,len(self.b1[0])))
                for w in range(self.nn_input_dim):
                    for u in range(len(self.W1[0])):
                        expmultsumdz2w1[w][u] = X[n][w]
                        expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)*dWnNextDtemp[w][u] #expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)
                        expmultsumdz2b1[0][u] = expmultsumdz2b1[0][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)*dWnNextDtemp[w][u] #np.exp(self.z2[n][t])
                        #mulz2fac = 0
                        #for t in range(self.nn_output_dim):
                        #    mulz2fac += np.exp(self.z2[n][t])*self.W2[u][t]
                        #expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*mulz2fac
                
                #create the final b derivative
                #expmultsumdz2b1 = np.ones((1,len(self.b1[0])))
                #for e in range(len(self.b1[0])):
                    #for u in range(len(self.W1[0])):
                #    for t in range(self.nn_input_dim):
                #        expmultsumdz2b1[0][e] = expmultsumdz2b1[0][e]*self.diff_actFun(self.z1[n][e],self.actFun_type)*dWnNextDtemp[e][t] #np.exp(self.z2[n][t])
            else:
                
                #create the multiplier factor for the derivative of the loss function
                expmultsumdz2wn = np.ones((len(self.W1),len(self.W1[0])))
                #expmultsumdz2w1
                #multiz2w = np.zeros((len(self.W1),len(self.W1[0])))
                #why are we iterating over the hidden dimension twice?
                #because the Wn is twice the hidden dimension rather than input and hidden like W1
                #loop through getting the output multiplication factor
                for w in range(self.nn_input_dim):
                    for u in range(len(self.W1[0])):
                        for t in range(self.nn_output_dim):
                            #create the exponential times the output weight factor
                            mulz2fac += np.exp(self.z2[n][t])*self.W2[u][t]
                        #stores values into a matrix to use for calculations
                        expmultsumdz2wn[w][u] = expmultsumdz2wn[w][u]*mulz2fac
                        
                        
                #create multiplication factor that multiplies by the previous layer derivative to multiply by the current layer derivative
                dWnNextDtemp = expmultsumdz2wn #np.ones((len(self.Wn[0]),len(self.Wn[0][0])))
                #dbnNextDtemp = expmultsumdz2wn[self.nn_layers_dim][0] #np.zeros((1,len(self.b1[0])))
                #expmultsumdz2bn = np.zeros((self.nn_layers_dim,len(self.b1[0])))
                #for v in range(self.nn_layers_dim-1):
                    
                    #dW1test is the derivative of the next layer, dWntest is the derivative of the current layer, dbntest is the derivative of the current layer b
                    #return dW1test, dWntest, dbntest
                #    dWnNextDeriv, dWnCurDeriv, dbnCurDeriv = self.Layer1.backprop(self.an[self.nn_layers_dim-v],self.Wn[self.nn_layers_dim-v],self.b1[self.nn_layers_dim-v-1],self.zn[self.nn_layers_dim-v-1], dWnNextDtemp) #backprop(self, X, W1, z1, hisderiv)
                    #loop through to create the bn multiplier
                #    for w in range(self.nn_hidden_dim):
                #        expmultsumdz2bn[self.nn_layers_dim-v-1][w] = dWnNextDeriv[w][u]*dbnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                        #loop through to create the wn multiplier
                #        for u in range(len(self.Wn[0][0][0])):                        
                #            expmultsumdz2wn[self.nn_layers_dim-v][w][u] = dWnNextDeriv[w][u]*dWnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                            
                #            dWnNextDtemp[w][u] = dWnNextDtemp[w][u]*dWnNextDeriv[w][u]
                            #expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type) #expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)
                            
                #create derivative variable for w1 matrix
                expmultsumdz2w1 = np.zeros((len(self.W1),len(self.W1[0])))
                for w in range(self.nn_input_dim):
                    for u in range(len(self.W1[0])):
                        expmultsumdz2w1[w][u] = X[n][w]
                        expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)*dWnNextDtemp[w][u] #expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)
                        #mulz2fac = 0
                        #for t in range(self.nn_output_dim):
                        #    mulz2fac += np.exp(self.z2[n][t])*self.W2[u][t]
                        #expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*mulz2fac
                
                #create the final b derivative
                expmultsumdz2b1 = np.zeros((1,len(self.b1[0])))
                for e in range(len(self.b1)):
                    #for u in range(len(self.W1[0])):
                    for t in range(self.nn_output_dim):
                        expmultsumdz2b1[0][e] += self.W2[e][t]*self.diff_actFun(self.z1[n][e],self.actFun_type)*np.exp(self.z2[n][t])
            
            #create the single derivative term w2
            derivz2w2 = np.zeros((len(self.W2),len(self.W2[0])))
            for w in range(len(self.W2)):
                for t in range(len(self.W2[0])):
                    derivz2w2[w][t] = self.a1[n][w]
            derivz2b2 =   np.ones((1,len(self.b2[0])))
            #for w in range(self.)
            #np.sum(self.a1[n], axis=0, keepdims=True)
            #print(derivz2w2)
            
            #create derivative
            
            deriva1b1 = np.zeros((self.nn_output_dim,len(self.b1[0])))
            for w in range(len(self.b1[0])):
                for u in range(len(self.W2[0])):
                    deriva1b1[u][w] = self.diff_actFun(self.z2[n][u],self.actFun_type)*self.W2[w][u]
            derivz2b1 = deriva1b1
            
            
            #create multiplication factor that multiplies by the previous layer derivative to multiply by the current layer derivative
            dWnNextDtemp = np.ones((len(self.Wn[0]), len(self.Wn[0][0]))) #expmultsumdz2wn[self.nn_layers_dim] #np.ones((len(self.Wn[0]),len(self.Wn[0][0])))
            #dbnNextDtemp = expmultsumdz2wn[self.nn_layers_dim][0] #np.zeros((1,len(self.b1[0])))
            expmultsumdz2bn2 = np.zeros((self.nn_layers_dim,len(self.b1[0])))
            derivz2bn = np.zeros((self.nn_layers_dim, len(self.bn[0]))) 
            deriva1wn = np.zeros((self.nn_layers_dim, len(self.Wn[0]),len(self.Wn[0][0]))) #np.sum(X[n], axis=0, keepdims=True)
            for p in range(self.nn_layers_dim):
            
                #dW1test is the derivative of the next layer, dWntest is the derivative of the current layer, dbntest is the derivative of the current layer b
                #def backprop(self, X, W1, z1, hisderiv):
                #return dW1test, dWntest, dbntest
                print(self.zn[0])
                dWnNextDeriv, dWnCurDeriv, dbnCurDeriv = self.Layer1.backprop(self.an[n][self.nn_layers_dim-p-1],self.Wn[self.nn_layers_dim-p-1],self.zn[n][self.nn_layers_dim-p-1], dWnNextDtemp) #backprop(self, X, W1, z1, hisderiv)
                #loop through to create the bn multiplier
                for w in range(self.nn_hidden_dim):
                    derivz2bn[self.nn_layers_dim-p-1][w] = dWnNextDeriv[w][u]*dbnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                    #loop through to create the wn multiplier
                    for u in range(len(self.Wn[0][0])):                        
                        deriva1wn[self.nn_layers_dim-p][w][u] = dWnNextDeriv[w][u]*dWnCurDeriv[w][u] #expmultsumdz2wn[self.nn_layers_dim-v][w][u]*
                        
                        dWnNextDtemp[w][u] = dWnNextDtemp[w][u]*dWnNextDeriv[w][u]
                        #expmultsumdz2w1[w][u] = expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type) #expmultsumdz2w1[w][u]*self.diff_actFun(self.z1[n][u],self.actFun_type)
                
            
                #for w in range(len(self.Wn)):
                #    for t in range(len(self.Wn[0])):
                #        for u in range(len(self.W2[0])):
                #            deriva1w1[u][w][t] = self.diff_actFun(self.z2[n][u],self.actFun_type)*self.W2[t][u]*X[n][w]
            derivz2wn = deriva1wn
            
            
            deriva1w1 = np.zeros((len(self.W2[0]),len(self.W1),len(self.W1[0]))) #np.sum(X[n], axis=0, keepdims=True)
            dWnNextDeriv, dWnCurDeriv, dbnCurDeriv = self.Layer1.backprop(self.a1, self.W1 ,self.b1[0], self.z1, dWnNextDtemp) #backprop(self, X, W1, z1, hisderiv)
            deriva1w1 = dWnCurDeriv
            #for w in range(len(self.W1)):
            #    for t in range(len(self.W1[0])):
            #        for u in range(len(self.W2[0])):
                        
                        
                        
            #            deriva1w1[u][w][t] = self.diff_actFun(self.z2[n][u],self.actFun_type)*self.W2[t][u]*X[n][w]
            derivz2w1 = deriva1w1
            #derivz2w1 = 0
            #derivz2b1 = 0 #np.zeros(self.nn)
            #derivz2w2 = [[self.a1[n]],[self.a1[n]]]
            #for k in range(self.nn_output_dim):
                
                #for b in range(self.nn_hidden_dim):
                #    derivz2w2 = self.a1[b]#np.sum(self.a1[n], axis=0, keepdims=True)
                #deriva1b1 = 1
                #derivz2b2 = 1
                #derivz2w1 = 0
                #for l in range(self.nn_input_dim):
                #    deriva1w1[l] = X[n][l] #np.sum(X[n], axis=0, keepdims=True)
                    
                    #derivz2b1 = 0
                #for b in range(self.nn_hidden_dim):
                   # derivz2w1[l] = self.W2[b][k]*self.diff_actFun(self.z1[n][b],self.actFun_type)*deriva1w1[l]
                    
                #derivz2b1 = self.W2[b][k]*self.diff_actFun(self.z1[n][b],self.actFun_type)*deriva1b1
                #print("a1")
                #print(self.a1)
                #print("b1")
                #print(self.b1)
                #print("b2")
                #print(self.b2)
                #print("exp_scores")
                #print(exp_scores)
                #print("exp_scores cell")
                #print(exp_scores[n][k])
                #print("derivz2w2[k]")
                #print(derivz2w2)
                #input()
                #for r in range(len(self.W2)):
                #    expmultsumdz2w2[r][k] = exp_scores[n][k]*derivz2w2[r]
                #for e in range(len(self.b2)):
                #expmultsumdz2b2[k] = exp_scores[n][k]#exp_scores[n][k]*derivz2b2
            #for k in range(len(self.W1))
                #for p in range(len(self.W1[0])):
                #    expmultsumdz2w1[k][p] = exp_scores[n][k]*self.W2[p][k]*self.diff_actFun(self.z1[p])*X[n][k] #derivz2w1
                #for o in range(len(self.b1)):
                #    expmultsumdz2b1[k] = exp_scores[n][k]*self.W2[o][k]*self.diff_actFun(self.z1[o])
                
            for i in range(self.nn_output_dim):
                #derivz2w2 = self.a1[n][i]
                #deriva1b1 = 1
                #derivz2b2 = 1
                #deriva1w1 = X[n][i]
                #derivz2w1 = self.W2[i][k]*self.diff_actFun(self.z1[n][i],self.actFun_type)*deriva1w1
                #derivz2b1 = self.W2[i][k]*self.diff_actFun(self.z1[n][i],self.actFun_type)*deriva1b1
                
                #expmultsum =                 
                dW2test[n] += (1/self.probs[n][i])*(derivz2w2[i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2w2) #dL/dW2
                db2test[n] += (1/self.probs[n][i])*(derivz2b2*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2b2) #dL/db2
                
                #for z in range(self.nn_layers_dim):
                #    dWntest[n][z] += (1/self.probs[n][i])*(derivz2w2[i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2wn[z]) #dL/dW2
                #print("derivz2w2[k]")
                #print(1/self.probs[n][i])
                #print("derivz2w2[k]")
                #print((derivz2w1[i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2w1))
                #input()
                dW1test[n] += (1/self.probs[n][i])*(derivz2w1[i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2w1) #dL/dW1
                db1test[n] += (1/self.probs[n][i])*(derivz2b1[i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2b1) #dL/db1
                for z in range(self.nn_layers_dim):
                    dWntest[n][z] += (1/self.probs[n][i])*(derivz2wn[z][i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2wn[z]) #dL/dW1
                    dbntest[n][z] += (1/self.probs[n][i])*(derivz2bn[z][i]*np.exp(self.z2[n][i])*(1/expsum[n])-np.exp(self.z2[n][i])*(1/(expsum[n]*expsum[n]))*expmultsumdz2bn[z]) #dL/db1
                
            dW2 += y[n]*dW2test[n]
            db2 += y[n]*db2test[n]
            dW1 += y[n]*dW1test[n]
            db1 += y[n]*db1test[n]
            dWn += y[n]*dWntest[n]
            dbn += y[n]*dbntest[n]
        
        dW2 = -dW2/num_examples#(1/self.probs)*(derivz2w2*exp(self.z2)*(1/expsum)-exp(self.z2)(1/(expsum*expsum))*expmultsum) #dL/dW2
        db2 = -db2/num_examples#(1/self.probs)*(derivz2b2*exp(self.z2)*(1/expsum)-exp(self.z2)(1/(expsum*expsum))*expmultsum) #dL/db2
        dW1 = -dW1/num_examples#(1/self.probs)*(derivz2w1*exp(self.z2)*(1/expsum)-exp(self.z2)(1/(expsum*expsum))*expmultsum) #dL/dW1
        db1 = -db1/num_examples#(1/self.probs)*(derivz2b1*exp(self.z2)*(1/expsum)-exp(self.z2)(1/(expsum*expsum))*expmultsum) #dL/db1
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
        #print("welll " + self.actFun_type)
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2, dWn, dbn = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            #print(self.reg_lambda)
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
                            #print("self.probs")
                            #print(self.probs)
                            input()
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


def main():
    # # generate and visualize Make-Moons dataset
     X, y = generate_data()
     #self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
     #network1 = NeuralNetwork(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
    #print(y)
     
     #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     #plt.show()
    #def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, nn_layers_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
     model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, nn_layers_dim=2, actFun_type='tanh')
     model.fit_model(X,y)
     model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()