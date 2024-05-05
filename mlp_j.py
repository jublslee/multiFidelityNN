import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as  transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from tqdm import tqdm
import numpy as np
import pandas as pd

# Multi-layer perceptron (MLP) model class for H1 layer (linear)
class MLP_H1(nn.Module):

    # constructor for the MLP model
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_H1, self).__init__()

        # Define hidden layer with "hidden_dim" linear neurons with input size "input_dim"
        self.W1 = torch.randn(input_dim, hidden_dim)

        # Define the output layer with "output_dim" linear neurons with input size "hidden_dim".
        self.W2 = torch.randn(hidden_dim, output_dim)

        # Define linear activation function
        self.linear = lambda x : x
        self.linear_prime = lambda x : 1

    # define the forward procedure for the network
    def forward(self, x):
        # Pass the input to the first layer
        self.z1 = torch.matmul(x, self.W1)

        # Apply the activation function in this first layer
        self.y1 = self.linear(self.z1)

        # Pass the output of the first layer to the next (output) layer
        self.z2 = torch.matmul(self.y1, self.W2)

        # Apply the activation function in the output layer
        y = self.linear(self.z2)

        return y

    # define the backward procedure for the network
    def backward(self, X, d_cost_d_y, y, learning_rate):

        d_cost_d_z2 = d_cost_d_y * self.linear_prime(self.z2)
        
        d_cost_d_y1 = torch.matmul(d_cost_d_z2, torch.t(self.W2)) 

        d_cost_d_z1 = d_cost_d_y1 * self.linear_prime(self.z1) 

        d_cost_d_W1 = torch.matmul(torch.t(X), d_cost_d_z1) 

        d_cost_d_W2 = torch.matmul(torch.t(self.y1), d_cost_d_z2) 

        self.W1 -= d_cost_d_W1*learning_rate
        self.W2 -= d_cost_d_W2*learning_rate

# Multi-layer perceptron (MLP) model class for H2 layer (nonlinear)
class MLP_H2(nn.Module):

    # constructor for the MLP model
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_H2, self).__init__()

        # Define hidden layer with "hidden_dim" linear neurons with input size "input_dim"
        self.W1 = torch.randn(input_dim, hidden_dim)
 
        # Define the ReLU activation function
        relu_zero = torch.tensor(0.0) # Define constant threshold
        self.ReLU = lambda x : torch.max(relu_zero, x)
        self.ReLU_prime = lambda x : torch.gt(x, relu_zero)

        # Define the output layer with "output_dim" linear neurons with input size "hidden_dim".
        self.W2 = torch.randn(hidden_dim, output_dim)

        # Define the sigmoid activation function
        sig_const = torch.tensor(-1.0) # define sigmoid constant a
        self.sigmoid = lambda x : (1/(1+torch.exp(sig_const*x)))
        self.sigmoid_prime = lambda x : self.sigmoid(x) * (1-self.sigmoid(x))

        # Define the tangent activation function 
        self.tanh = lambda x : ((torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x)))
        self.tanh_prime = lambda x : 1-self.tanh(x)**2

        # Define the SoftPlus activation function (middle layer)
        self.soft = lambda x : (torch.log(1 + torch.exp(x)))
        self.soft_prime = lambda x : self.sigmoid(x)

        # Define parametric ReLU
        self.PReLU = lambda x : torch.where(x < 0.0, 0.01 * x, x)
        self.PReLU_prime = lambda x : torch.where(x < 0.0, 0.01, 1)

    # define the forward procedure for the network
    def forward(self, x):
        # Pass the input to the first layer
        self.z1 = torch.matmul(x, self.W1)

        # Apply the activation function in this first layer
        self.y1 = self.ReLU(self.z1)

        # Pass the output of the first layer to the next (output) layer
        self.z2 = torch.matmul(self.y1, self.W2)

        # Apply the activation function in the output layer
        y = self.sigmoid(self.z2)

        return y

    # define the backward procedure for the network
    def backward(self, X, d_cost_d_y, y, learning_rate):

        d_cost_d_z2 = d_cost_d_y * self.sigmoid_prime(self.z2)
        
        # d_cost_d_y1 = torch.matmul(d_cost_d_z2, torch.t(self.W2)) 
        d_cost_d_y1 = torch.matmul(torch.t(d_cost_d_z2), (self.W2)) 

        d_cost_d_z1 = d_cost_d_y1 * self.ReLU_prime(self.z1) 

        d_cost_d_W1 = torch.matmul(torch.t(X), d_cost_d_z1) 

        d_cost_d_W2 = torch.matmul(torch.t(self.y1), d_cost_d_z2) 

        self.W1 -= d_cost_d_W1*learning_rate
        self.W2 -= d_cost_d_W2*learning_rate