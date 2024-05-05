import torch
import numpy as np
import matplotlib.pyplot as plt

# @torch.jit.script
class HF_Problem():
    def __init__(self):
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]
    
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return (6*x - 2)**2 * torch.sin(12*x - 4)
    
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return 12*(6*x - 2) * torch.sin(12*x - 4) + 12*(6*x - 2)**2 * torch.cos(12*x - 4)

def lf_mk1(x: torch.Tensor) -> torch.Tensor:
    return 0.5*(6*x - 2)**2 * torch.sin(12*x - 4) + 10*(x - 0.5) - 5

def lf_mk1_grad(x: torch.Tensor) -> torch.Tensor:
    return 6*(6*x - 2)*(torch.sin(12*x - 4) + (6*x - 2)*torch.cos(12*x - 4))+10

class LF_Problem():
    def __init__(self):
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]
        pass
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return lf_mk1(x)
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return lf_mk1_grad(x)
    

def test_case():

    HF = HF_Problem()
    LF = LF_Problem()

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