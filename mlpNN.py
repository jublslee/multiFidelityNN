import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as  transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.functional import resize
from torchvision.transforms import CenterCrop
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from mlp import *

torch.set_default_dtype(torch.float)

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mlpNN(nn.Module):

    def __init__(self, input_size, output_size, layers, activations, init_type, pe=None): #, input_mask=None):

        # Construct parent class
        super().__init__()

        # Set activations
        if(activations == 'ReLU'):
            self.act = nn.ReLU()
        elif(activations == 'SiLU'):
            self.act = nn.SiLU()
        elif(activations == 'Tanh'):            
            self.act = nn.Tanh()
        elif(activations == 'Identity'):            
            self.act = nn.Identity()
        else:
            print('ERROR: Invalid MLP activation.')
            exit(-1)

        # Store positional encoding 
        self.pe = pe

        self.linears = nn.ModuleList()
        for i in range(len(layers)+1):
            if(i == 0):
                in_dim  = input_size
                out_dim = layers[i]
            elif(i == len(layers)):
                in_dim  = layers[i-1]
                out_dim = output_size
            else:
                in_dim  = layers[i-1]
                out_dim = layers[i]

            # Create new linear layer
            if((i == 0) and (self.pe is not None)):
                self.linears.append(torch.nn.Linear(self.pe.get_num_features(), out_dim, dtype=torch.float))
            else:
                self.linears.append(torch.nn.Linear(in_dim, out_dim, dtype=torch.float))

            # Init layer
            if(init_type == 'Xavier normal'):
                nn.init.xavier_normal_(self.linears[-1].weight)
            elif(init_type == 'Xavier uniform'):
                nn.init.xavier_uniform_(self.linears[-1].weight)
            elif(init_type == 'Glorot normal'):
                nn.init.glorot_normal_(self.linears[-1].weight)
            elif(init_type == 'Glorot uniform'):
                nn.init.glorot_uniform_(self.linears[-1].weight)
            else:
                print('ERROR: Invalid initialization type.')
                exit(-1)            
            # Init bias
            self.linears[-1].bias.data.fill_(0.0)

    def forward(self, inputs, debug=False):

        if(self.pe is not None):
            x = self.pe(inputs)
        else:
            x = inputs

        if(debug): print(x.size())

        for linear in self.linears[:-1]:
            x = (self.act(linear(x)))
            if(debug): print(x.size())

        # Last layer without activation
        x = self.linears[-1](x)
        if(debug): print(x.size())

        return x