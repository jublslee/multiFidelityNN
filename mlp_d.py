import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from nrf.transformation import mlp_transformation
# from nrf.networks.posenc import PositionalEncoding
# from nrf.experiment import Experiment

torch.set_default_dtype(torch.float)
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):

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

        # Store inputs mask
        # self.input_mask = input_mask

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
            # Weight masking - To be implemented!!!
            # if j==0
            #     self.wfx.weight.data.mul_(self.mask_use)
            x = (self.act(linear(x)))
            if(debug): print(x.size())

        # Last layer without activation
        x = self.linears[-1](x)
        if(debug): print(x.size())

        return x
    
# class mlp_net(nn.Module):

#     def __init__(self,exp,io_size,norm=None):
#         super().__init__()

#         # Set Transformation
#         if(norm is not None):
#             x_norm  = (norm['x_n1'],norm['x_n2'])
#             u_norm  = (norm['u_n1'],norm['u_n2'])
#             self._input_transform = mlp_transformation('input',x_norm,scale_type=exp.data_normalization)
#             self._output_transform = mlp_transformation('output',u_norm,scale_type=exp.data_normalization)
#         else:
#             self._input_transform = None
#             self._output_transform = None

#         # Create POSITIONAL ENCODING
#         if(exp.use_pe == True):
#             self.pe = PositionalEncoding(num_feat = io_size['input'], 
#                                          num_freq = exp.pe_num_freq, 
#                                          pe_type = exp.pe_type, 
#                                          bandwidth = exp.pe_bandwidth,
#                                          feat_lin=exp.pe_feat_lin)
#         else:
#             self.pe = None

#         # Create MLP
#         self.mlp = MLP(io_size['input'], io_size['output'], exp.mlp_branch_layers, exp.mlp_activations, exp.mlp_init_type, pe=self.pe).to(device)

#     def forward(self, x, debug=False):

#         # Transform inputs
#         if(self._input_transform is not None):
#             x = self._input_transform(x)

#         # Eval MLP
#         x = self.mlp(x)

#         # Transform outputs
#         if(self._output_transform is not None):
#             x = self._output_transform(x)

#         # Return
#         return x   
    
def test_mlp():
    
    # Set parameters
    input_size = 3
    # input_size = (4,2)
    output_size = 3
    layers = [64,64]
    activations = 'ReLU'
    init_type = 'Xavier normal'

    # Create input
    # if we input pairs (x,y), what is going to be the output? 
    # is it also going to be a pair?
    x = torch.rand((3,))
    print(x)

    # Create network
    net =  MLP(input_size, output_size, layers, activations, init_type)

    # Evaluate
    y_pred = net(x) # ,debug=True)

    print(y_pred)

# def test_mlp_net():

#     exp = Experiment()

#     # Size of inputs and outputs
#     io_size = {}
#     io_size['input']  = 4
#     io_size['output'] = 1

#     # Create inputs
#     x = torch.tensor([[0.5,0.5,0.5,1.0]])

#     # Normalization constants
#     norms = {}
#     norms['x_n1'] = torch.tensor([0.0,0.0,0.0,0.0])
#     norms['x_n2'] = torch.tensor([1.0,1.0,1.0,1.0])
#     norms['u_n1']  = torch.tensor([0.0])
#     norms['u_n2']  = torch.tensor([1.0])

#     # Create network 
#     net = mlp_net(exp,io_size,norms)
  
#     # Evaluate network
#     out = net(x,debug=True)

#     # Print output 
#     print(out)

# TEST NETWORK
if __name__ == "__main__":

    # Test network
    test_mlp()
    # test_mlp_net()

   
