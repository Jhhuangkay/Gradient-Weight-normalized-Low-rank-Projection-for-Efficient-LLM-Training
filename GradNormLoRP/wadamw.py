import math
import torch
from typing import Callable, Iterable, Tuple
from torch import nn
from torch.optim import Optimizer
from .weight_norm_galore import WeightNormGaLore


class WeightNormGaLoreAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        rank: int = 256,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = 'std',
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        print("Parameter groups and their tensor sizes:")
        
        # for group_idx, group in enumerate(self.param_groups):
        #     print(f"\nParameter group {group_idx + 1}:")
        #     for param_idx, param in enumerate(group["params"]):
        #         print(f"  Tensor {param_idx + 1} size: {param.size()}")
                
        for group in self.param_groups:
            print()
            for p in group["params"]:
                print('0'*50)
                print(f"p's size is : {p.shape}")
                print('0'*50)
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    print('x'*50)
                    print(f"grad's size is : {grad.shape}")
                    print(f"p's size is : {p.shape}")
                    print('x'*50)
                    state["weight_norm_galore"] = WeightNormGaLore(
                        p.clone().detach(), rank=group["rank"], update_proj_gap=group["update_proj_gap"],
                        scale=group["scale"], proj_type=group["proj_type"], device=p.device
                    )
                    state["direction_optimizer"] = torch.optim.Adam([state["weight_norm_galore"].direction], lr=group["lr"])
                    state["magnitude_optimizer"] = torch.optim.Adam([state["weight_norm_galore"].magnitude], lr=group["lr"])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Compute low-rank projected gradients
                print(f"Step {state['step']}: Computing low-rank projected gradients.")
                grad_direction, grad_magnitude = state["weight_norm_galore"].compute_gradients(grad)
                projected_grad = state["weight_norm_galore"].project_gradients(grad_direction, state["step"])

                 # Ensure grad_magnitude has the correct shape
                if grad_magnitude.numel() != state["weight_norm_galore"].magnitude.numel():
                    raise ValueError(f"Mismatch in number of elements: grad_magnitude {grad_magnitude.shape} vs magnitude {state['weight_norm_galore'].magnitude.shape}")

                if grad_magnitude.shape != state["weight_norm_galore"].magnitude.shape:
                    grad_magnitude = grad_magnitude.view_as(state["weight_norm_galore"].magnitude)
                
                # Update direction and magnitude
                print(f"Step {state['step']}: Updating direction.")
                state["direction_optimizer"].zero_grad()
                state["weight_norm_galore"].direction.grad = projected_grad
                state["direction_optimizer"].step()
                
                print(f"Step {state['step']}: Updating magnitude.")
                state["magnitude_optimizer"].zero_grad()
                state["weight_norm_galore"].magnitude.grad = grad_magnitude
                state["magnitude_optimizer"].step()

                # Reconstruct weight
                # print(f"Step {state['step']}: Reconstructing weight.")
                # state["weight_norm_galore"].reconstruct_weight()

                # Apply weight decay at the end
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

                torch.cuda.empty_cache()
                # # Debugging: Print shapes to ensure consistency
                # print(f"Decomposing weight again to check shapes")
                # direction, magnitude = state["weight_norm_galore"].weight_norm_decompose(p)
                # print(f"W shape: {p.shape}")
                # print(f"Direction shape: {direction.shape}")
                # print(f"Magnitude shape: {magnitude.shape}")

        return loss
