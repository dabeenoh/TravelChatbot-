import numpy as np
import pandas as pd

import re
import heapq
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import json

MAX_NB_WORDS = 256

# Math activation functions
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

# forward propagation
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

# backward propagation to calculate gradient
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
    # number of examples
    m = A_prev.shape[1]

    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)

    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

# update the values using the previous parameters and the gradient
def update(params_values, grads_values, nn_architecture, learning_rate):
    """Gradient descent"""
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

# Preprocessing of text (questions asked to the chatbot)
def _clean_text(text):
    """Returns a lemmatized and cleaned text from raw data
       Remove special chars and stopwords (common word without meaning like "the")"""
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.

    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text
# Vectorize the function so that it is numpy compatible
v_clean_text = np.vectorize(_clean_text)

class LoadingData():
    """Load data into a pandas DataFrame and preprocess it"""
    def __init__(self,test_size = 0.18,random_state = 42,verbose=0):
        filename = 'data/intents.json'
        json_file = json.load(open(filename))
        # Explode the list of questions of the dataframe
        self.data_frame = pd.DataFrame(json_file['intents'])[['tag','patterns']].explode('patterns')
        # X and Y extracted from intent.json
        self.X = np.array(self.data_frame['patterns'])
        self.Y = pd.get_dummies(self.data_frame['tag']).values
        if verbose==1:
            print('Shape of label tensor:', self.Y.shape)
        # Decoding an output vector into an intent label and then into a response TODO: make a method
        Y_to_label = np.sort(self.data_frame['tag'].unique())
        self.cat_to_tag = dict(enumerate(Y_to_label))
        self.tag_to_cat = {value:key for (key,value) in self.cat_to_tag.items()}
        df_tag_to_response = pd.DataFrame(json_file['intents'])[['tag','responses']]
        self.tag_to_response = dict(df_tag_to_response.set_index('tag')['responses'])
        # Train - Test split
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y, test_size = test_size, random_state = random_state)

class DNN():
    """Deep Neural Network for multiclass classification"""
    def __init__(self):
        self.VOCAB = {}
        self.nn_architecture = None
        # parameters storage initiation
        self.params_values = {}
        # initiation of lists storing the history
        # of metrics calculated during the learning process
        self.cost_history = []
        self.accuracy_history = []
        self._type = "DNN_scratch"

    def init_layers(self, nn_architecture, seed = 99):
        # random seed initiation
        np.random.seed(seed)

        # iteration over network layers
        for idx, layer in enumerate(nn_architecture):
            # we number network layers from 1
            layer_idx = idx + 1

            # extracting the number of units in layers
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            # initiating the values of the W matrix
            # and vector b for subsequent layers
            self.params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            self.params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

    def get_cost_value(self,Y_hat, Y):
        """Categorical Crossentropy because classification problem"""
        # number of examples
        m = Y_hat.shape[1]
        # calculation of the cost according to the formula
        log_likelihood = -np.log(Y_hat+1e-9)
        cost = np.sum(Y*log_likelihood) / m
        return np.squeeze(cost)

    def get_accuracy_value(self,Y_hat, Y):
        Y_hat_ = convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def build(self,X_train,Y_train):
        # Builds a dictionnary of most used words (BAG OF WORDS) into self.VOCAB
        wordfreq = {}
        corpus = v_clean_text(np.array(X_train))
        for sentence in corpus:
            tokens = word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1
        self.VOCAB = heapq.nlargest(MAX_NB_WORDS, wordfreq, key=wordfreq.get)
        self.nn_architecture = [
            {"input_dim": len(self.VOCAB), "output_dim": 128, "activation": "relu"},
            {"input_dim": 128, "output_dim": 64, "activation": "relu"},
            {"input_dim": 64, "output_dim": 32, "activation": "relu"},
            {"input_dim": 32, "output_dim": 16, "activation": "relu"},
            {"input_dim": 16, "output_dim": Y_train.shape[1], "activation": "sigmoid"},
        ]
        # initiation of neural net parameters
        self.init_layers(self.nn_architecture, 2)

    def _bagging(self,corpus):
        sentence_vectors = []
        for sentence in corpus:
            sentence_tokens = word_tokenize(sentence)
            sent_vec = []
            for token in self.VOCAB:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
        return np.asarray(sentence_vectors)

    def get_input_array(self,sentences):
        corpus = v_clean_text(np.array(sentences))
        return self._bagging(corpus)

    def train(self, X_train, Y_train, epochs=10000, learning_rate=0.01, verbose=False, callback=None):
        # performing calculations for subsequent iterations
        X,Y = np.transpose(X_train),np.transpose(Y_train)
        for i in range(epochs):
            # step forward
            Y_hat, cashe = full_forward_propagation(X, self.params_values, self.nn_architecture)

            # calculating metrics and saving them in history
            cost = self.get_cost_value(Y_hat, Y)
            self.cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            self.accuracy_history.append(accuracy)

            # step backward - calculating gradient
            grads_values = full_backward_propagation(Y_hat, Y, cashe, self.params_values, self.nn_architecture)
            # updating model state
            self.params_values = update(self.params_values, grads_values, self.nn_architecture, learning_rate)

            if(i % 500 == 0):
                if(verbose):
                    print("Iteration: %.5f - crossentropy loss: %.5f - accuracy: %.5f"%(i, cost, accuracy))
                if(callback is not None):
                    callback(i, self.params_values)

    def predict(self,X):
        Y_pred = full_forward_propagation(np.transpose(X), self.params_values, self.nn_architecture)[0]
        return np.transpose(Y_pred)

    def evaluate(self,Y_pred,Y_real):
        Y_hat, Y = np.transpose(Y_pred), np.transpose(Y_real)
        loss = self.get_cost_value(Y_hat, Y)
        accuracy = self.get_accuracy_value(Y_hat, Y)
        return loss, accuracy

    def save(self):
        with open("model/scratch.h5",'wb') as output_file:
            pickle.dump(self.params_values,output_file)

    def load(self,path="model/scratch.h5"):
        with open("model/scratch.h5",'rb') as input_file:
            self.params_values = pickle.load(input_file)
