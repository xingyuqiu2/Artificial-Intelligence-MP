# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size  #d = 3072
        self.out_size = out_size
        
        self.hidden_size = 32    #h = 32
        self.lin1 = nn.Linear(in_size, self.hidden_size)
        self.lin2 = nn.Linear(self.hidden_size, out_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = self.lrate)
        #raise NotImplementedError("You need to write this part!")

    def set_parameters(self, params):
        """ Sets the parameters of your network.

        @param params: a list of tensors containing all parameters of the network
        """
        self.params = params
        #raise NotImplementedError("You need to write this part!")
    
    def get_parameters(self):
        """ Gets the parameters of your network.

        @return params: a list of tensors containing all parameters of the network
        """
        return self.params
        #raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        N = x.shape[0]
        y = torch.ones(N, self.out_size)
        for i in range(N):
            hidden = self.lin1(x[i])
            output = self.lin2(self.relu(hidden))
            y[i] = self.sigmoid(output)
        return y
        #raise NotImplementedError("You need to write this part!")
        #return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        self.optimizer.zero_grad()
        # Forward pass
        y_hat = self.forward(x)
        # Compute loss
        loss = self.loss_fn(y_hat, y)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()
        #raise NotImplementedError("You need to write this part!")
        #return 0.0


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    train_set = (train_set - torch.mean(train_set)) / torch.std(train_set)
    N = train_set.shape[0]
    M = dev_set.shape
    losses = []
    yhats = torch.ones(M, dtype=int)

    model = NeuralNet(0.07, nn.CrossEntropyLoss(), 3072, 2)
    model.train()

    train_data = []
    for i in range(N):
        train_data.append([train_set[i], train_labels[i]])
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True)

    for i in range(n_iter):
        x, y = iter(loader).next()
        loss = model.step(x, y)
        losses.append(loss)
    
    y_hats = model.forward(dev_set)
    for i in range(len(y_hats)):
        yhats[i] = torch.argmax(y_hats[i])
    yhats = np.array(yhats)
    return losses, yhats, model
    #raise NotImplementedError("You need to write this part!")
    #return [],[],None
