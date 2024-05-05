######### Import Packages #########
import torch
import torch.nn as nn

# 32 bit Floating Point
# torch.set_default_dtype(torch.float)
# 64 bit Floating Point
torch.set_default_dtype(torch.double)

# Seed the random number generator for all devices (both CPU and CUDA)
torch.manual_seed(0)

#################################################################################

#### deepNN_mlp: definition for deep neural networks skeleton ####
## feed forward deep NN
## accepts uniform number of neuron in each hidden layer
class deepNN(nn.Module):
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
    def __init__(self, in_dim,
                #  num_hidden_layers,
                 out_dim,
                 hidden_param,
                #  num_neurons: int,
                 add_bias=True,
                 activation='Tanh'):
        
        super().__init__()

        # Define parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_layer = hidden_param[0]
        neuron_size = hidden_param[1]

        # Set activations
        if(activation == 'ReLU'):
            self.act_fn = nn.ReLU()
        else: # use Tanh
            self.act_fn = nn.Tanh()
        
        # Create sequential order of layers
        self.layer_order = []
        self.weights = []

        # Define input layer
        self.layer_order.append(nn.Linear(self.in_dim, neuron_size))

        # Define activation function
        self.layer_order.append(self.act_fn) 

        # Define hidden layer & activation function
        for layer in range(hidden_layer):
            self.layer_order.append(nn.Linear(neuron_size, neuron_size))
            self.layer_order.append(self.act_fn) 

        # Define output layer
        self.layer_order.append(nn.Linear(neuron_size, self.out_dim,
                                          bias = add_bias))
        
        # Combine layers using nn.Sequential
        self.net = nn.Sequential(*self.layer_order)

        for layer in self.layer_order:
            if isinstance(layer, nn.Linear):
                self.weights.append(layer.weight)
    
    def initialize(self, initialize_func):
        for k, layer in enumerate(self.layer_order):
            if k % 2 == 0:
              initialize_func(layer.weight)
              if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def NNet_test():
    nnet = deepNN(3, 3, [3,30])
    nnet.initialize(torch.nn.init.zeros_)

    x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    # print("Weights of each layer:")
    # for layer in nnet.layer_order:
    #     if isinstance(layer, nn.Linear):
    #         print(layer.weight)

    # print("Output after forward pass:")
    print(nnet(x))

# TESTING CLASS
if __name__ == "__main__":

    # NNet definition test
    NNet_test()

    