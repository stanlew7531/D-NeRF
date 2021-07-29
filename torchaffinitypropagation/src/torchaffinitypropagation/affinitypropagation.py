from typing import Optional
import torch

from torchaffinitypropagation.cpu import affinitypropagation_cpu

def affinity_propagation(similarities: torch.Tensor, iterations: int):
    print(similarities)
    responsibilities, availabilities = affinitypropagation_cpu(similarities, iterations)

    return responsibilities, availabilities