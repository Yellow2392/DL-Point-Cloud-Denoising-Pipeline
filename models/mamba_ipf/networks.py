import torch
import torch.nn as nn
from .backbone import FeatureExtraction, resolve_mamba_class

# Dos instancias independientes del mismo backbone, aplicadas en secuencia:
#   - MambaOutlierClassifier (fase 1, poda): un logit por punto -> máscara de outliers.
#   - MambaDisplacementRegressor (fase 2, suavizado): desplazamiento 3D por punto, refinado de forma
#     iterativa con num_modules redes encadenadas (esquema IterativePFN / 3DMambaIPF).
# No comparten pesos: cada fase ve una distribución de entrada distinta (con y sin outliers) y optimiza
# un objetivo distinto, por lo que compartir el tronco forzaría un compromiso entre ambas tareas.

class MambaOutlierClassifier(nn.Module):
    kind = "classifier"

    def __init__(self, mamba_cls, k_neighbors=32):
        super().__init__()
        self.net = FeatureExtraction(mamba_cls, k=k_neighbors, output_dim=1, output_activation=None)

    def forward(self, x):
        # x: (B, N, 3) normalizado -> logits (B, N)
        return self.net(x).squeeze(-1)

class MambaDisplacementRegressor(nn.Module):
    kind = "regressor"

    def __init__(self, mamba_cls, k_neighbors=32, num_modules=4):
        super().__init__()
        # Mismo nombre que en DenoiseNet original ("feature_nets") para poder cargar sus pesos
        self.feature_nets = nn.ModuleList([
            FeatureExtraction(mamba_cls, k=k_neighbors, output_dim=3, output_activation="tanh")
            for _ in range(num_modules)
        ])
        # Capa de salida en cero: sin entrenar, el regresor es la identidad (desplazamiento nulo) en lugar
        # de mover los puntos al azar. Se sobrescribe si se cargan pesos preentrenados.
        for net in self.feature_nets:
            nn.init.zeros_(net.linear3.weight)
            nn.init.zeros_(net.linear3.bias)

    def forward(self, x):
        # Devuelve la entrada y el desplazamiento predicho de cada módulo (necesarios para la pérdida)
        inputs, displacements = [], []
        for net in self.feature_nets:
            displacement = net(x)
            inputs.append(x)
            displacements.append(displacement)
            x = x + displacement
        return inputs, displacements

    def denoise(self, x):
        inputs, displacements = self(x)
        return inputs[-1] + displacements[-1]

MODEL_CLASSES = {"classifier": MambaOutlierClassifier, "regressor": MambaDisplacementRegressor}

def build_model(kind, model_args, device, mamba_impl="auto"):
    mamba_cls = resolve_mamba_class(mamba_impl, device)
    return MODEL_CLASSES[kind](mamba_cls, **model_args).to(device)

def save_checkpoint(path, model, model_args, patch_size, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": model.kind,
        "model_args": model_args,
        "patch_size": patch_size,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }, path)

def load_checkpoint(path, device, mamba_impl="auto"):
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["kind"], checkpoint["model_args"], device, mamba_impl)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint

def load_pretrained_ipf(model, path):
    # Inicializa desde un checkpoint del 3DMambaIPF original (Lightning, entrenado en PUNet).
    # Regresor: se cargan sus feature_nets tal cual. Clasificador: el backbone del primer módulo,
    # excepto la capa de salida (3 canales en el original, 1 aquí). Devuelve las claves cargadas.
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    if model.kind == "classifier":
        prefix = "feature_nets.0."
        state = {"net." + k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)
