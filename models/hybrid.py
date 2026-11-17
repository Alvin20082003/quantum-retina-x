import torch
import torch.nn as nn
from transformers import ViTModel
from .quantum import QuantumLayer

class QuantumRetinaX(nn.Module):
    """ViT (frozen, pretrained Phase 2) -> reducer -> VQC -> classifier."""

    def __init__(self, cfg, vit_weights_path="weights/vit_baseline.pth", num_classes=None):
        super().__init__()
        nc = num_classes or cfg["data"]["num_classes"]
        nq = cfg["quantum"]["n_qubits"]
        nl = cfg["quantum"]["n_layers"]
        std = cfg["quantum"]["init_std"]

        # Load the ViT backbone with Phase 2 trained weights
        self.backbone = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k")
        hidden = self.backbone.config.hidden_size  # 768

        # Load Phase 2 weights into backbone (only backbone part)
        if vit_weights_path:
            state = torch.load(vit_weights_path, weights_only=True,
                               map_location="cpu")
            backbone_state = {k.replace("backbone.", ""): v
                              for k, v in state.items() if k.startswith("backbone.")}
            self.backbone.load_state_dict(backbone_state)
            print(f"[hybrid] loaded ViT backbone from {vit_weights_path}")

        # FREEZE the backbone - it's already good from Phase 2
        for p in self.backbone.parameters():
            p.requires_grad = False
        print(f"[hybrid] backbone frozen ({sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M params)")

        # These are the ONLY trainable layers:
        self.reducer = nn.Linear(hidden, nq)     # 768 -> 4
        self.vqc = QuantumLayer(n_qubits=nq, n_layers=nl, init_std=std)
        self.head = nn.Linear(nq, nc)             # 4 -> 5

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[hybrid] trainable params: {trainable} (reducer + VQC + head)")

    def forward(self, x, return_features=False):
        # No torch.no_grad() here: the backbone's own params are frozen
        # (requires_grad=False above) so they never get optimizer updates
        # either way, but keeping the graph connected lets Grad-CAM compute
        # gradients at backbone layers when the caller marks x as requiring
        # grad (e.g. backend/main.py's explainability path).
        out = self.backbone(pixel_values=x)
        cls_token = out.last_hidden_state[:, 0]
        reduced = self.reducer(cls_token)
        quantum_out = self.vqc(reduced)
        logits = self.head(quantum_out)
        if return_features:
            return logits, cls_token
        return logits
