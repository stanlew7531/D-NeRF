import torch
from torchaffinitypropagation.cpu import affinitypropagation_cpu

sim_matrix = torch.Tensor([[-22,-7,-6,-12,-17],\
                             [-7,-22,-17,-17,-22],\
                             [-6,-17,-22,-18,-21],\
                             [-12,-17,-18,-22,-3],\
                             [-17,-22,-21,-3,-22]])

if __name__ == "__main__":
    resp, avail = affinitypropagation_cpu(sim_matrix, 100)
    print(resp)
    print(avail)