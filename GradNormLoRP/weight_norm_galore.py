import torch
from torch import nn
from torch.optim import Optimizer
import time
from .galore_projector import GaLoreProjector

class WeightNormGaLore(nn.Module):
    def __init__(self, W, rank, update_proj_gap=200, scale=1.0, proj_type='std', device='cpu'):
        super(WeightNormGaLore, self).__init__()
        self.device = device
        self.W = W.to(device)
        self.direction, self.magnitude = self.weight_norm_decompose(W)
        self.direction = nn.Parameter(self.direction.detach().clone().to(device), requires_grad=True)
        self.magnitude = nn.Parameter(self.magnitude.detach().clone().to(device), requires_grad=True)
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        
        # Keep reference to original weight
        self.original_weight = self.W
        self.original_weight.weight_norm_galore_param = True

        # # Mark these parameters as part of WeightNormGaLore
        # self.direction.weight_norm_galore_param = True
        # self.magnitude.weight_norm_galore_param = True
        
        

        self.projector = GaLoreProjector(rank, update_proj_gap=update_proj_gap, scale=scale, proj_type=proj_type)

    def weight_norm_decompose(self, W):
        print("Decomposing weight")
        epsilon = 1e-12
        if W.dim() == 1:
            magnitude = torch.norm(W, dim=0, keepdim=True).to(self.device) + epsilon
            direction = (W / magnitude).to(self.device)
        else:
            magnitude = torch.norm(W, dim=1, keepdim=True).to(self.device) + epsilon
            direction = (W / magnitude).to(self.device)
        # Print shapes for debugging
        print(f"W shape: {W.shape}")
        print(f"Direction shape: {direction.shape}")
        print(f"Magnitude shape: {magnitude.shape}")
        return direction, magnitude
        
        

    def compute_gradients(self, grad_W):
        print("Computing gradients")
        grad_magnitude = (self.direction * grad_W).sum(dim=1).to(self.device)
        grad_direction = (grad_W - self.direction * grad_magnitude.unsqueeze(1)).to(self.device)
        
        # Print shapes for debugging
        print(f"Grad direction shape: {grad_direction.shape}")
        print(f"Grad magnitude shape: {grad_magnitude.shape}")
        return grad_direction, grad_magnitude

    def project_gradients(self, grad_direction, step):
        print(f"Projecting gradients at step {step}")
        start_time = time.time()
        G_low_rank = self.projector.project(grad_direction.to(self.device), step)
        print(f"Projection forward took {time.time() - start_time:.4f} seconds")
        projected_grad = self.projector.project_back(G_low_rank).to(self.device)
        print(f"Projection took {time.time() - start_time:.4f} seconds")
        print(f"Projection backward took {time.time() - start_time:.4f} seconds")
        return projected_grad

    def reconstruct_weight(self):
        print("Reconstructing weight")
        with torch.no_grad():
            print(f"Direction shape: {self.direction.shape}")
            print(f"Magnitude shape: {self.magnitude.shape}")
            self.W.copy_((self.direction * self.magnitude.unsqueeze(1)).to(self.device))
            print(f"Reconstructed W shape: {self.W.shape}")
            
            
