######### Import Packages #########
import torch
import torch.nn as nn
from deepNN import *

#### MLP ####
## network with low-fidelity, linear & non-linear high-fidelity architecture
## network introduced in Karniadakis paper
class MLP(nn.Module):
    ### init: initialize ####
    ## @param in_dim 
    ##        input dimension (int)
    ## @param out_dim
    ##        output dimension (int)
    ## @param hidden_param
    ##        parameters for hidden layer [number of layers, number of neuron size]
    ## @param add_bias 
    ##        choice of adding bias 
    ## @param activation
    ##        type of activation function to use
    def __init__(self,
                 lf_list: list,
                 hf,
                 lf_nn_sizes: set,
                 nl_nn_size: set):
        """
        Initialize neural network models

        Parameters
        ----------
        model_list : list
            Models part of multi-fidelity network.

        
        Returns
        -------
        Wrapper for multi-fidelity neural network.
        """
        super().__init__()

        lf_num = len(lf_list)

        # Low-fidelity surrogates
        self.lf_models = nn.ModuleList()
        sum_lf_out_dim = 0
        for lf_i in range(lf_num):
            self.lf_models.append(deepNN(lf_list[lf_i].in_dim,
                                       lf_list[lf_i].out_dim,
                                       [lf_nn_sizes[lf_i]['num_hidden_layers'], lf_nn_sizes[lf_i]['num_neurons']])
                                       )
            sum_lf_out_dim += lf_list[lf_i].out_dim

        # High-fidelity correlation network
        # Linear correlation subnetwork
        # Linear neural network (no hidden layers, no activation functions)
        self.hf_l = nn.Linear(hf.in_dim + sum_lf_out_dim,
                              hf.out_dim)
        # Non-linear correlation subnetwork
        self.hf_nl = deepNN(hf.in_dim + sum_lf_out_dim,
                          hf.out_dim,
                          [nl_nn_size['num_hidden_layers'], nl_nn_size['num_neurons']],
                          add_bias=False)

        # Apply Xavier uniform initialization to all networks
        for lf_model in self.lf_models:
            lf_model.initialize(torch.nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.hf_l.weight)
        self.hf_nl.initialize(torch.nn.init.xavier_uniform_)
        self.lf_weights_biases = []
        # self.lf_weight = 
        # self.lf_bias = 

    def get_lf_weights_biases(self):
        # lf_weights_biases = []
        for lf_model in self.lf_models:
            self.lf_weights_biases.append((lf_model.weight, lf_model.bias))
        # return lf_weights_biases

    # def get_hf_weights_biases(self):
    #     hf_weights_biases = [(self.hf_l.weight, self.hf_l.bias), (self.hf_nl.weight, self.hf_nl.bias)]
    #     return hf_weights_biases
    
    def eval(self):
        """
        Sets all models to evaluation mode.
        """
        # for lf in self.lfs:
        for lf in self.lf_models:
            lf.eval()
        self.hf_l.eval()
        self.hf_nl.eval()

    def hf_prediction(self, x):
        num_lf_models = len(self.lf_models)

        yL = [self.lf_models[kk](x) for kk in range(num_lf_models)]
        hf_in = torch.cat((x, *yL), dim=1)
        hf_l_pred = self.hf_l(hf_in)
        hf_nl_pred = self.hf_nl(hf_in)
        
        return hf_l_pred + hf_nl_pred

def test_MLP():
    from exCases import HF_MK1, LF_MK1

    # Define models for LF
    lf_list = [LF_MK1()]

    num_lf_models = len(lf_list)
    lf_nn_sizes = [{ 'num_hidden_layers': 2, 'num_neurons': 10, }
                   for ii in range(num_lf_models)]
    nl_nn_size = {
        'num_hidden_layers': 2,
        'num_neurons': 10,
    }

    mfnn = MLP(lf_list, HF_MK1(), lf_nn_sizes, nl_nn_size)

    x = torch.Tensor([[3],[4],[5]])
    
    print(mfnn.hf_prediction(x))

# TESTING CLASS
if __name__ == "__main__":
    test_MLP()
    