######### Import Packages #########
import torch
import numpy as np
import matplotlib.pyplot as plt
#################################################################################

#### LF_Problem: class defining the low fidelity problem ####
class LF_Problem():
    def __init__(self):
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]

    # LF problem solution
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return 0.5*(6*x - 2)**2 * torch.sin(12*x - 4) + 10*(x - 0.5) - 5
    
    # LF problem derivative solution
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return 6*(6*x - 2)*(torch.sin(12*x - 4) + (6*x - 2)*torch.cos(12*x - 4))+10
    
#### HF_Problem: class defining the high fidelity problem ####
class HF_Problem():
    def __init__(self):
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]
    
    # HF problem solution
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return (6*x - 2)**2 * torch.sin(12*x - 4)
    
    # HF problem derivative solution
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return 12*(6*x - 2) * torch.sin(12*x - 4) + 12*(6*x - 2)**2 * torch.cos(12*x - 4)
    

def test_case():
    # Construct problem
    HF = HF_Problem()
    LF = LF_Problem()
    
    # Set up parameters
    x  = torch.linspace(0.0,1.0,100)
    lf = HF.eval(x)
    hf = LF.eval(x)
    
    # Plot for both low & high fidelity to check if it works as expected
    plt.plot(x.numpy(),lf.numpy(), label='Low-Fidelity')
    plt.plot(x.numpy(),hf.numpy(), label='High-Fidelity')
    plt.legend() # display labels
    plt.show() # show resulting plots

# Define main function for testing
if __name__ == "__main__":
    test_case()