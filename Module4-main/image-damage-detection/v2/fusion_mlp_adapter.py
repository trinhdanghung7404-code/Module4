import os, torch, numpy as np
import torch.nn as nn

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "damage_fusion_mlp.pt")

class DamageFusionMLP(nn.Module):
    def __init__(self, in_features=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

_mlp_model = None

def get_mlp_adapter():
    global _mlp_model
    if _mlp_model is None:
        _mlp_model = DamageFusionMLP(in_features=14)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location="cpu")
            _mlp_model.load_state_dict(state)
        _mlp_model.eval()
    return _mlp_model

def extract_triangle_vector(t: dict):
    # DINOv2 metrics
    dino_sim = float(t.get("dino_sim", 1.0))
    dino_min = float(t.get("dino_min_sim", dino_sim))
    dino_l2 = float(np.sqrt(max(0.0, 2.0 - 2.0 * dino_sim)))
    dino_gap = float(dino_sim - dino_min)
    
    # Layer 1 Structural metrics
    dark_crack = float(t.get("dark_crack", 0.0))
    white_scratch = float(t.get("white_scratch", 0.0))
    intrusive = float(t.get("intrusive_len", 0.0))
    blob = float(t.get("blob_area", 0.0))
    z_l1 = float(t.get("z_l1", 0.0))
    ssim_val = float(t.get("ssim", 1.0))
    desc_sim = float(t.get("desc_sim", 1.0))
    ori_corr = float(t.get("ori_corr", 1.0))
    
    # Layer 2 Color metrics
    chroma_err = float(t.get("chroma_err", 0.0))
    has_cb = 1.0 if t.get("has_color_blob", False) else 0.0
    
    return [
        dino_sim, dino_min, dino_l2, dino_gap,
        dark_crack, white_scratch, intrusive, blob,
        z_l1, ssim_val, desc_sim, ori_corr,
        chroma_err, has_cb
    ]

def predict_triangle_defect_probability(t: dict) -> float:
    """Computes defect probability using the trained Fusion MLP Adapter."""
    vec = extract_triangle_vector(t)
    model = get_mlp_adapter()
    with torch.no_grad():
        x = torch.tensor([vec], dtype=torch.float32)
        logit = model(x)
        prob = float(torch.sigmoid(logit).item())
    return prob
