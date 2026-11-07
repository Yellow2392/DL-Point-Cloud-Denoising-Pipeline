import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone de 3DMambaIPF (Zhou et al., AAAI 2025; basado en IterativePFN): convoluciones de aristas
# dinámicas (DGCNN) intercaladas con bloques Mamba. Adaptado de temp/3DMambaIPF/models/{feature,block,
# dynamic_edge_conv}.py con estos cambios:
#   - EdgeConv en PyTorch puro (sin torch_geometric) y kNN con torch.cdist (sin pytorch3d).
#   - Sin las ramas fused_add_norm / RMSNorm de Triton ni la proyección z_dim (no se usaban).
#   - Cabeza de salida configurable: 3 canales + tanh (desplazamiento) o 1 canal sin activación (logit).
# Los nombres de los parámetros se conservan, de modo que los pesos preentrenados originales se pueden cargar.

def resolve_mamba_class(impl, device):
    # "cuda": kernels de mamba_ssm (requiere GPU NVIDIA); "reference": PyTorch puro (CPU, lento);
    # "auto": kernels de mamba_ssm si hay GPU y el paquete está instalado, si no la referencia.
    if impl not in ("auto", "cuda", "reference"):
        raise ValueError("MAMBA_IMPL debe ser 'auto', 'cuda' o 'reference'.")
    if impl != "reference" and torch.device(device).type == "cuda":
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            return Mamba
        except ImportError:
            if impl == "cuda":
                raise
    elif impl == "cuda":
        raise RuntimeError("MAMBA_IMPL='cuda' requiere ejecutar en un dispositivo CUDA.")
    from .mamba_fallback import Mamba
    return Mamba

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    # Inicialización de mamba_ssm / GPT-2 (reescalado de las proyecciones de salida por 1/sqrt(n_layer))
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class Block(nn.Module):
    # Add -> LayerNorm -> Mixer, devolviendo la salida del mixer y el residuo (estructura de mamba_ssm)
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, drop_path=0.0):
        super().__init__()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states, residual=None):
        residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        return self.mixer(hidden_states), residual

class MixerModel(nn.Module):
    def __init__(self, d_model, n_layer, mamba_cls, norm_epsilon=1e-5, drop_out_in_block=0.0, drop_path=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(
                d_model,
                partial(mamba_cls, layer_idx=i),
                norm_cls=partial(nn.LayerNorm, eps=norm_epsilon),
                drop_path=drop_path,
            )
            for i in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.apply(partial(_init_weights, n_layer=n_layer))
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
            hidden_states = self.drop_out_in_block(hidden_states)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

def knn_indices(x, k):
    # x: (B, N, C) -> índices (B, N, k) de los k vecinos más cercanos de cada punto, sin el propio punto
    dist = torch.cdist(x, x)
    return dist.topk(k + 1, dim=-1, largest=False).indices[..., 1:]

class DynamicEdgeConv(nn.Module):
    # EdgeConv con agregación máxima: h_i = max_j MLP([x_i, x_j - x_i]) + Lin(x_i)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(),
        )
        self.lin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU())

    def forward(self, x, knn_idx):
        # x: (B, N, C); knn_idx: (B, N, k)
        batch, n_points, channels = x.shape
        k = knn_idx.shape[-1]
        flat_idx = (knn_idx + torch.arange(batch, device=x.device).view(-1, 1, 1) * n_points).reshape(-1)
        x_j = x.reshape(batch * n_points, channels)[flat_idx].view(batch, n_points, k, channels)
        x_i = x.unsqueeze(2).expand(-1, -1, k, -1)
        edges = torch.cat([x_i, x_j - x_i], dim=-1).reshape(-1, 2 * channels)
        messages = self.mlp(edges).view(batch, n_points, k, self.out_channels).max(dim=2).values
        return messages + self.lin(x.reshape(-1, channels)).view(batch, n_points, self.out_channels)

class FeatureExtraction(nn.Module):
    # Red por punto de 3DMambaIPF. Los puntos del parche deben venir ordenados por distancia a la semilla:
    # Mamba procesa la nube como una secuencia, así que ese orden forma parte de la entrada.
    def __init__(self, mamba_cls, k=32, embedding_dim=512, output_dim=3, output_activation="tanh"):
        super().__init__()
        self.k = k
        self.output_activation = output_activation

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)
        self.conv3 = DynamicEdgeConv(48, 144)
        self.conv4 = DynamicEdgeConv(16 + 48 + 144, embedding_dim)

        self.mamba_1 = MixerModel(16, 6, mamba_cls)
        self.mamba_2 = MixerModel(48, 6, mamba_cls)
        self.mamba_3 = MixerModel(144, 6, mamba_cls)
        self.mamba_4 = MixerModel(embedding_dim, 1, mamba_cls)
        self.mamba_5 = MixerModel(256, 1, mamba_cls)
        self.mamba_6 = MixerModel(128, 1, mamba_cls)

        self.linear1 = nn.Linear(embedding_dim, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: (B, N, 3) -> (B, N, output_dim)
        x1 = self.mamba_1(self.conv1(x, knn_indices(x, self.k)))
        x2 = self.mamba_2(self.conv2(x1, knn_indices(x1, self.k)))
        x3 = self.mamba_3(self.conv3(x2, knn_indices(x2, self.k)))
        x = self.conv4(torch.cat((x1, x2, x3), dim=-1), knn_indices(x3, self.k))
        x = self.mamba_4(x)
        x = F.relu(self.mamba_5(self.linear1(x)))
        x = F.relu(self.mamba_6(self.linear2(x)))
        x = self.linear3(x)
        return torch.tanh(x) if self.output_activation == "tanh" else x
