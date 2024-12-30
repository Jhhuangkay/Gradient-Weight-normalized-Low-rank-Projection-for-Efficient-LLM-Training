from bitsandbytes.optim.optimizer import Optimizer2State
import torch
from .weight_norm_galore import WeightNormGaLore  # Import the decoupled class

class WeightNormGaLoreAdamW8bit(Optimizer2State):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, optim_bits=32, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False):
        super().__init__("adam", params, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                    state["weight_norm_galore"] = WeightNormGaLore(
                        p, group["rank"], group["update_proj_gap"],
                        group["scale"], group["proj_type"], device=p.device
                    )
                    state["direction_optimizer"] = torch.optim.Adam([state["weight_norm_galore"].direction], lr=group["lr"])
                    state["magnitude_optimizer"] = torch.optim.Adam([state["weight_norm_galore"].magnitude], lr=group["lr"])

                grad = p.grad
                state["step"] += 1

                # GaLore Projection for weight_norm_galore
                grad_direction, grad_magnitude = state["weight_norm_galore"].compute_gradients(grad)
                G_low_rank = state["weight_norm_galore"].project_gradients(grad_direction, state["step"])

                # Update direction and magnitude
                state["direction_optimizer"].zero_grad()
                state["weight_norm_galore"].direction.grad = G_low_rank
                state["direction_optimizer"].step()

                state["magnitude_optimizer"].zero_grad()
                state["weight_norm_galore"].magnitude.grad = grad_magnitude
                state["magnitude_optimizer"].step()

                # Reconstruct weight
                state["weight_norm_galore"].reconstruct_weight()
                p.data.copy_(state["weight_norm_galore"].get_weights())

                if 'state1' not in state:
                    self.init_state(group, p, gindex, pindex)

                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                torch.cuda.synchronize()

        if self.is_paged:
            torch.cuda.synchronize()

        return loss
