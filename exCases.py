import torch
import numpy as np
import matplotlib.pyplot as plt

# @torch.jit.script
class HF_MK1():
    def __init__(self):
        self.model_name = 'HF-MK1'
        self.model_type = 'PhysModel'
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]
        # pass9
    
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return (6*x - 2)**2 * torch.sin(12*x - 4)
    
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return 12*(6*x - 2) * torch.sin(12*x - 4) + 12*(6*x - 2)**2 * torch.cos(12*x - 4)

# @torch.jit.script
def lf_mk1(x: torch.Tensor) -> torch.Tensor:
    return 0.5*(6*x - 2)**2 * torch.sin(12*x - 4) + 10*(x - 0.5) - 5
# @torch.jit.script
def lf_mk1_grad(x: torch.Tensor) -> torch.Tensor:
    return 6*(6*x - 2)*(torch.sin(12*x - 4) + (6*x - 2)*torch.cos(12*x - 4))+10

# @torch.jit.script
class LF_MK1():
    def __init__(self):
        self.model_name = 'LF-MK1'
        self.model_type = 'PhysModel'
        self.in_dim     = 1
        self.out_dim    = 1
        self.limits     = [(0.0,1.0)]
        pass
    def eval(self,x: torch.Tensor) -> torch.Tensor:
        return lf_mk1(x)
    def eval_grad(self,x: torch.Tensor) -> torch.Tensor:
        return lf_mk1_grad(x)
    

def test_case():

    HF = HF_MK1()
    LF = LF_MK1()

    x  = torch.linspace(0.0,1.0,100)
    lf = HF.eval(x)
    hf = LF.eval(x)

    plt.plot(x.numpy(),hf.numpy())
    plt.plot(x.numpy(),lf.numpy())
    plt.show()

# TESTING CLASS
if __name__ == "__main__":
    test_case()