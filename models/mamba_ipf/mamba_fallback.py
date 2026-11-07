import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementación de referencia en PyTorch puro del bloque Mamba (SSM selectivo), equivalente al camino
# lento de mamba_ssm.modules.mamba_simple.Mamba (v1.1.x) y con los mismos nombres y formas de parámetros:
# un checkpoint entrenado con los kernels CUDA de mamba_ssm se carga aquí sin conversión.
# Es mucho más lenta (el escaneo recorre la secuencia en un bucle de Python); está pensada para inferencia
# en equipos sin CUDA y para pruebas, no para entrenar.

def selective_scan(u, delta, A, B, C, D):
    # u, delta: (B, L, D); A: (D, N); B, C: (B, L, N); D: (D,)
    batch, seqlen, d_inner = u.shape
    state = u.new_zeros(batch, d_inner, A.shape[1])
    ys = []
    for i in range(seqlen):
        delta_i = delta[:, i].unsqueeze(-1)                                   # (B, D, 1)
        state = torch.exp(delta_i * A) * state + delta_i * u[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1)
        ys.append(torch.einsum("bdn,bn->bd", state, C[:, i]))
    return torch.stack(ys, dim=1) + u * D

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False,
                 use_fast_path=True, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=conv_bias, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Inicialización de dt idéntica a mamba_ssm
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        seqlen = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)                       # (B, L, D) cada uno
        x = self.act(self.conv1d(x.transpose(1, 2))[..., :seqlen]).transpose(1, 2)  # convolución causal
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                                       # incluye delta_bias
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, delta, A, B, C, self.D)
        return self.out_proj(y * self.act(z))
