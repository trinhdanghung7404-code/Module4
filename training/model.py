"""
training/model.py
=================
Định nghĩa kiến trúc mạng nơ-ron CeramicDefectHead (CD-MLP).
Kiến trúc: 3-Layer Dense Bottleneck MLP (384 -> 128 -> 32 -> 4)
Tối ưu hóa siêu nhẹ (~53,860 tham số, ~0.21 MB), huấn luyện cực nhanh trên CPU.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

CLASS_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "Crack",
    2: "Chip",
    3: "Stain"
}

CLASS_NAMES_VI: Dict[int, str] = {
    0: "Men lanh / Binh thuong",
    1: "Vet nut / Ran vo",
    2: "Vet sut me",
    3: "Vet o ban / Lem mau"
}


class CeramicDefectHead(nn.Module):
    """
    Tầng nơ-ron phân loại khuyết tật gốm sứ nhận đầu vào là vector đặc trưng DINOv2 (384 chiều).
    Phân loại thành 4 lớp:
      - 0: Normal (Vùng men nguyên vẹn / Lóa sáng nhẹ)
      - 1: Crack (Vết rạn nứt kết cấu)
      - 2: Chip (Vết sứt mẻ vành miệng / đế bình)
      - 3: Stain (Vết ố vàng / loang mực)
    """

    def __init__(self, in_features: int = 384, num_classes: int = 4, dropout: float = 0.2):
        super(CeramicDefectHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Tầng 1: Chiếu giảm chiều và chuẩn hóa phân phối (384 -> 128)
        self.layer1 = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        # Tầng 2: Cô đọng đặc tính khuyết tật vật lý (128 -> 32)
        self.layer2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(p=dropout / 2.0)
        )

        # Tầng 3: Phân loại đầu ra (32 -> 4)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: Tensor (B, 384)
        Đầu ra: Logits (B, 4)
        """
        feat = self.layer1(x)
        feat = self.layer2(feat)
        logits = self.classifier(feat)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dự đoán lớp và xác suất tin cậy Softmax.
        Trả về: (predicted_class_ids, confidence_scores)
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)
        return preds, confs

    def count_parameters(self) -> int:
        """Tính tổng số tham số có thể huấn luyện"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, path: str):
        """Lưu trọng số mô hình"""
        torch.save({
            "model_state_dict": self.state_dict(),
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "class_names": CLASS_NAMES,
            "class_names_vi": CLASS_NAMES_VI
        }, path)

    def load_weights(self, path: str, device: str = "cpu"):
        """Nạp trọng số từ file .pt"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
        self.to(device)
        self.eval()
