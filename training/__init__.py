"""
training/__init__.py
====================
Mô-đun độc lập phục vụ sinh dữ liệu, trích xuất đặc trưng và huấn luyện mô hình CeramicDefectHead.
Tách biệt hoàn toàn, không ảnh hưởng đến luồng code chính của hệ thống.
"""

from .model import CeramicDefectHead, CLASS_NAMES, CLASS_NAMES_VI

__all__ = ["CeramicDefectHead", "CLASS_NAMES", "CLASS_NAMES_VI"]
