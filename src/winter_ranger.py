# WinterRanger optimizer
#
# Inspired by RangerLite, a refactored/unbloated Ranger21 derivative used by Stockfish:
#   - RangerLite by @TonyCongqianWang
#   - Ranger21  by @lessw2020  (https://github.com/lessw2020/Ranger21)
#
# The numerical core follows RangerLite: Positive-Negative Momentum (PNM), per-parameter
# Adam second moment, Stable Weight Decay (decay scaled by the global gradient variance),
# and Norm Loss (pulls each unit's weight-vector norm toward 1). Adapted to fit this
# codebase:
#   * Lookahead is a periodic merge into the live weights (every lookahead_k steps) rather
#     than a fast/slow pointer swap, so validation, checkpointing and engine serialization
#     always see the averaged weights with no optimizer.eval()/train() calls required.
#   * Resume-safe: the global Lookahead counter is stored in state_dict().
#   * The Ranger21 "legacy scoping bug" compatibility path is omitted.
# Every component can be toggled off for ablation.

import math
import torch


class WinterRanger(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-7,
                 pnm=True, pnm_momentum=1.0, normloss=True, normloss_factor=1e-4,
                 lookahead=True, lookahead_k=5, lookahead_alpha=0.5):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
                        pnm_momentum=pnm_momentum, normloss_factor=normloss_factor)
        super().__init__(params, defaults)
        self.pnm = pnm
        self.normloss = normloss
        self.lookahead = lookahead
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.lookahead_step = 0
        self.eps = eps
        self.param_size = 0  # cached on the first step; used by Stable Weight Decay

    @staticmethod
    def _unit_norm(x):
        """L2 norm of each sub-unit (neuron/filter), keepdim, for the norm-loss term."""
        if x.ndim <= 1:
            return x.norm(p=2.0)
        return x.norm(dim=tuple(range(1, x.ndim)), keepdim=True, p=2.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        first_param = next((p for g in self.param_groups for p in g["params"] if p.grad is not None), None)
        if first_param is None:
            return loss

        # Phase 1: update each parameter's second moment and accumulate the global,
        # debiased variance sum that Stable Weight Decay normalises by.
        variance_ma_sum = torch.zeros(1, device=first_param.device)
        param_size = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("WinterRanger does not support sparse gradients")
                param_size += p.numel()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["grad_ma"] = torch.zeros_like(p)
                    state["variance_ma"] = torch.zeros_like(p)
                    if self.pnm:
                        state["neg_grad_ma"] = torch.zeros_like(p)
                    if self.lookahead:
                        state["lookahead_params"] = p.detach().clone()
                state["step"] += 1
                bias_correction2 = 1 - beta2 ** state["step"]
                variance_ma = state["variance_ma"]
                variance_ma.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                variance_ma_sum += (variance_ma / bias_correction2).sum()

        if not self.param_size:
            self.param_size = param_size
        variance_normalized = torch.sqrt(variance_ma_sum / self.param_size).clamp_min(self.eps)

        # Phase 2: stable weight decay, norm loss, and the PNM/Adam update.
        for group in self.param_groups:
            lr = group["lr"]
            decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            pnm_factor = group["pnm_momentum"]
            normloss_factor = group["normloss_factor"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                step = state["step"]

                if decay:
                    p.mul_(1 - decay * lr / variance_normalized)
                if self.normloss:
                    unorm = self._unit_norm(p)
                    corr = 2 * normloss_factor * (1 - (unorm + eps).reciprocal())
                    p.mul_(1 - lr * corr)

                bias_correction2 = 1 - beta2 ** step
                denom = (state["variance_ma"].sqrt() / math.sqrt(bias_correction2)).add_(eps)

                if self.pnm:
                    # Positive-Negative Momentum: two alternating first-moment buffers.
                    if step % 2 == 1:
                        pos_ma, neg_ma = state["grad_ma"], state["neg_grad_ma"]
                    else:
                        pos_ma, neg_ma = state["neg_grad_ma"], state["grad_ma"]
                    pos_ma.mul_(beta1 ** 2).add_(p.grad, alpha=1 - beta1 ** 2)
                    effective_step = ((step + 1) // 2) * 2
                    bias_correction1 = 1 - beta1 ** effective_step
                    noise_norm = math.sqrt((1 + pnm_factor) ** 2 + pnm_factor ** 2)
                    update = pos_ma.mul(1 + pnm_factor).add(neg_ma, alpha=-pnm_factor).mul(1 / noise_norm)
                    step_size = lr / bias_correction1
                    p.addcdiv_(update, denom, value=-step_size)
                else:
                    bias_correction1 = 1 - beta1 ** step
                    grad_ma = state["grad_ma"]
                    grad_ma.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                    step_size = lr / bias_correction1
                    p.addcdiv_(grad_ma, denom, value=-step_size)

        # Lookahead: every k steps pull the live weights toward the slow copy and resync.
        # The averaged result stays in p.data, so save/serialize/validate use it directly.
        if self.lookahead:
            self.lookahead_step += 1
            if self.lookahead_step >= self.lookahead_k:
                self.lookahead_step = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        state = self.state[p]
                        if "lookahead_params" not in state:
                            continue
                        slow = state["lookahead_params"]
                        p.mul_(self.lookahead_alpha).add_(slow, alpha=1 - self.lookahead_alpha)
                        slow.copy_(p)
        return loss

    def state_dict(self):
        d = super().state_dict()
        d["lookahead_step"] = self.lookahead_step
        return d

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.lookahead_step = state_dict.pop("lookahead_step", 0)
        super().load_state_dict(state_dict)
