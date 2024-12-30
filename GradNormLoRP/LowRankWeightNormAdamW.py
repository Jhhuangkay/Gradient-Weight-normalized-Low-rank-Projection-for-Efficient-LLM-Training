# This implementation of LowRankWeightNormAdamW integrates the GaLore projector to perform gradient projection
# and projection-back during optimization. The GaLore projector is based on the work described in:
# "GaLore: Efficient Training of Large Models with Low-Rank Projections."
# If this integration is used in your work, please cite the GaLore paper accordingly.
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .galore_projector import GaLoreProjector

class LowRankWeightNormAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        rank: int = None,  # Default to None if not using low-rank projection
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std',
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        # Include additional parameters in defaults
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            rank = group.get('rank', None)
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                if p.data.dim() == 2:
                    norm_dim = 1
                elif p.data.dim() == 1:
                    norm_dim = 0
                else:
                    raise ValueError("Unsupported tensor dimension for weight normalization")

                # State initialization
                if "step" not in state:
                    state['step'] = 0

                    # Initialize weight normalization parameters and their averages based on the dimensions of p
                    if rank is not None:
                        state['projector'] = GaLoreProjector(rank, update_proj_gap=group['update_proj_gap'], scale=group['scale'], proj_type=group['proj_type'])
                         # Initialize exp_avg and exp_avg_sq after projection
                        
                    state['v'] = p.data.clone()
                    state['g'] = torch.norm(state['v'], p=2, dim=norm_dim, keepdim=True)

                # Apply weight normalization
                norm_v = torch.norm(state['v'], p=2, dim=norm_dim, keepdim=True)
                state['g'].data = norm_v  # Update to reflect the latest magnitude
                normalized_weight = state['g'] * (state['v'] / norm_v)
                p.data.copy_(normalized_weight)  # Update p with the normalized weight

                # Compute gradients for v and g
                grad_v = grad * state['g'] / norm_v
                grad_g = torch.sum(grad * (state['v'] / norm_v), dim=norm_dim, keepdim=True)

                # Project gradient to low-rank space for v if rank is specified
                if rank is not None:
                    #print(f"Before projection: grad_v.shape={grad_v.shape}")
                    grad_v = state['projector'].project(grad_v, state['step'])
                    #print(f"After projection: grad_v.shape={grad_v.shape}")

                if "exp_avg_v" not in state:
                    state['exp_avg_v'] = torch.zeros_like(grad_v)
                    state['exp_avg_sq_v'] = torch.zeros_like(grad_v)
                if "exp_avg_g" not in state:
                    state['exp_avg_g'] = torch.zeros_like(state['g'])
                    state['exp_avg_sq_g'] = torch.zeros_like(state['g'])

                exp_avg_v, exp_avg_sq_v = state['exp_avg_v'], state['exp_avg_sq_v']
                exp_avg_g, exp_avg_sq_g = state['exp_avg_g'], state['exp_avg_sq_g']
                beta1, beta2 = group['betas']

                state['step'] += 1
                

                # Decay the first and second moment running average coefficient for v
                exp_avg_v.mul_(beta1).add_(grad_v, alpha=1 - beta1)
                exp_avg_sq_v.mul_(beta2).addcmul_(grad_v, grad_v, value=1 - beta2)
                denom_v = exp_avg_sq_v.sqrt().add_(group['eps'])

                # Decay the first and second moment running average coefficient for g
                exp_avg_g.mul_(beta1).add_(grad_g, alpha=1 - beta1)
                exp_avg_sq_g.mul_(beta2).addcmul_(grad_g, grad_g, value=1 - beta2)
                denom_g = exp_avg_sq_g.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Compute norm gradient for v
                norm_grad_v = exp_avg_v / denom_v

                # Compute norm gradient for g
                norm_grad_g = exp_avg_g / denom_g

                # Project back from low-rank space for v if rank is specified
                if rank is not None:
                    #print(f"Before project back: norm_grad_v.shape={norm_grad_v.shape}")
                    norm_grad_v = state['projector'].project_back(norm_grad_v)
                    #print(f"After project back: norm_grad_v.shape={norm_grad_v.shape}")

                # Update v parameters
                state['v'].add_(norm_grad_v, alpha=-step_size)

                # Update g parameters
                state['g'].add_(norm_grad_g, alpha=-step_size)

                # Apply weight decay if needed
                if group['weight_decay'] != 0:
                    state['v'].add_(state['v'], alpha=-group['lr'] * group['weight_decay'])
                    state['g'].add_(state['g'], alpha=-group['lr'] * group['weight_decay'])

                # Reconstruct the weight and update the original parameter
                norm_v = torch.norm(state['v'], p=2, dim=norm_dim, keepdim=True)
                p.data = state['g'] * (state['v'] / norm_v)

        return loss