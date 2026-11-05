"""
training/train.py
=================
Quy trình huấn luyện Tầng nơ-ron phân loại khuyết tật gốm sứ (CeramicDefectHead).
Tự động:
1. Nạp hoặc tự sinh dataset_features.pt qua Sổ địa chỉ
2. Phân chia 80% Train / 20% Validation
3. Huấn luyện với AdamW + Cosine Annealing Scheduler (25 Epochs)
4. Tự động lưu trọng số vào training/defect_head.pt
5. Tự động vẽ 3 biểu đồ chuẩn học thuật phục vụ Báo cáo Đồ án Chương 3:
   - training/figures/loss_curve.png
   - training/figures/accuracy_curve.png
   - training/figures/confusion_matrix.png
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Chế độ vẽ không cần GUI window
import matplotlib.pyplot as plt

# Đảm bảo in tiếng Việt không bị lỗi cp1252 trên Windows console
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Đảm bảo import được module local
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model import CeramicDefectHead, CLASS_NAMES, CLASS_NAMES_VI


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Tính ma trận nhầm lẫn (Confusion Matrix) thuần NumPy không cần scikit-learn"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_academic_figures(
    history: dict,
    cm: np.ndarray,
    figures_dir: str
):
    """Vẽ 3 biểu đồ phục vụ báo cáo đồ án"""
    os.makedirs(figures_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Biểu đồ Loss Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, history["train_loss"], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(epochs, history["val_loss"], 'r--s', label='Validation Loss', linewidth=2, markersize=4)
    plt.title('DINOv2 + CD-MLP Loss Convergence Curve', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Cross-Entropy Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    loss_path = os.path.join(figures_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()
    print(f"[FIGURE] Đã lưu biểu đồ Loss tại: {loss_path}")

    # 2. Biểu đồ Accuracy Curve
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], 'g-o', label='Training Accuracy', linewidth=2, markersize=4)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], 'm--^', label='Validation Accuracy', linewidth=2, markersize=4)
    plt.title('Classification Accuracy over Epochs', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.ylim([60, 102])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    acc_path = os.path.join(figures_dir, "accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()
    print(f"[FIGURE] Đã lưu biểu đồ Accuracy tại: {acc_path}")

    # 3. Biểu đồ Confusion Matrix
    plt.figure(figsize=(7, 6), dpi=300)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix (4 Classes)', fontsize=13, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(4)
    labels = [f"{CLASS_NAMES[i]}" for i in range(4)]
    plt.xticks(tick_marks, labels, rotation=25, fontsize=10)
    plt.yticks(tick_marks, labels, fontsize=10)

    # Điền giá trị % và số lượng mẫu vào từng ô
    thresh = cm_norm.max() / 2.0
    for i in range(4):
        for j in range(4):
            val_pct = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"
            color = "white" if cm_norm[i, j] > thresh else "black"
            plt.text(j, i, val_pct, horizontalalignment="center", verticalalignment="center",
                     color=color, fontsize=9, fontweight='bold')

    plt.ylabel('Ground Truth Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[FIGURE] Đã lưu ma trận nhầm lẫn tại: {cm_path}")


def train_pipeline(
    pt_dataset_path: str,
    output_model_path: str,
    figures_dir: str,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3
):
    print("=" * 65)
    print(" KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN CERAMIC DEFECT MLP HEAD")
    print("=" * 65)
    start_time = time.time()

    # Nạp dữ liệu
    print(f"[DATA] Đang đọc tập dữ liệu: {pt_dataset_path}")
    data = torch.load(pt_dataset_path, weights_only=False)
    features = data["features"]  # (N, 384)
    labels = data["labels"]      # (N,)
    num_samples = len(labels)

    # Phân chia 80% Train, 20% Val có bảo toàn tỷ lệ nhãn (Stratified)
    indices = np.arange(num_samples)
    train_indices = []
    val_indices = []

    for c in range(4):
        c_idxs = indices[labels.numpy() == c]
        np.random.shuffle(c_idxs)
        n_train = int(len(c_idxs) * 0.8)
        train_indices.extend(c_idxs[:n_train])
        val_indices.extend(c_idxs[n_train:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    x_train, y_train = features[train_indices], labels[train_indices]
    x_val, y_val = features[val_indices], labels[val_indices]

    print(f"[DATA] Phân chia dữ liệu:")
    print(f"  • Tập huấn luyện (Train 80%): {len(x_train)} mẫu")
    print(f"  • Tập kiểm thử (Validation 20%): {len(x_val)} mẫu")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # Khởi tạo mô hình
    device = torch.device("cpu")
    model = CeramicDefectHead(in_features=384, num_classes=4, dropout=0.2).to(device)
    print(f"[MODEL] Tổng số tham số có thể huấn luyện: {model.count_parameters():,} tham số")

    # Hàm mất mát & Tối ưu hóa
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print("\n" + "-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc (%)':^15} | {'Val Loss':^12} | {'Val Acc (%)':^13}")
    print("-" * 65)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 1. Huấn luyện (Train)
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        scheduler.step()

        train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # 2. Kiểm định (Validation)
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)

                total_val_loss += loss.item() * bx.size(0)
                preds = torch.argmax(logits, dim=-1)
                correct_val += (preds == by).sum().item()
                total_val += bx.size(0)

                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(by.cpu().numpy())

        val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc*100:^15.2f} | {val_loss:^12.4f} | {val_acc*100:^13.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(output_model_path)

    elapsed_time = time.time() - start_time
    print("-" * 65)
    print(f"\n[DONE] Hoàn tất huấn luyện trong {elapsed_time:.2f} giây!")
    print(f"[DONE] Độ chính xác kiểm định cao nhất (Best Val Acc): {best_val_acc*100:.2f}%")
    print(f"[DONE] Trọng số mô hình đã lưu tại: {output_model_path}")

    # Đồng bộ 1 bản copy sang models/defect_head.pt
    base_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(base_dir, "models")
    if os.path.exists(models_dir):
        import shutil
        target_pt = os.path.join(models_dir, "defect_head.pt")
        shutil.copy2(output_model_path, target_pt)
        print(f"[SYNC] Đã lưu bản sao trọng số tại: {target_pt}")

    # Tính Confusion Matrix trên tập kiểm định
    cm = compute_confusion_matrix(np.array(val_targets_list), np.array(val_preds_list), num_classes=4)

    # Xuất đồ thị học thuật
    plot_academic_figures(history, cm, figures_dir)

    # In báo cáo bảng học thuật Chương 3
    print("\n" + "=" * 65)
    print(" BẢNG TỔNG HỢP THÔNG SỐ HUẤN LUYỆN (BÁO CÁO ĐỒ ÁN CHƯƠNG 3)")
    print("=" * 65)
    print(f"| {'Hạng mục':<30} | {'Giá trị':<28} |")
    print(f"|{'-'*32}|{'-'*30}|")
    print(f"| {'Tên mô hình':<30} | {'DINOv2-DefectNet (CD-MLP)':<28} |")
    print(f"| {'Mô hình nền tảng (Backbone)':<30} | {'DINOv2 (dinov2_vits14 - Frozen)':<28} |")
    print(f"| {'Kiến trúc tầng học (Head)':<30} | {'Dense 384 -> 128 -> 32 -> 4':<28} |")
    print(f"| {'Số tham số huấn luyện':<30} | {f'{model.count_parameters():,} params':<28} |")
    print(f"| {'Dung lượng tệp trọng số':<30} | {'~0.21 MB':<28} |")
    print(f"| {'Thuật toán tối ưu':<30} | {'AdamW (weight_decay=0.01)':<28} |")
    print(f"| {'Tốc độ học ban đầu':<30} | {'1e-3 (Cosine Annealing)':<28} |")
    print(f"| {'Số Epoch':<30} | {f'{epochs} Epochs':<28} |")
    print(f"| {'Thời gian huấn luyện CPU':<30} | {f'{elapsed_time:.1f} giây':<28} |")
    print(f"| {'Độ chính xác Validation':<30} | {f'{best_val_acc*100:.2f}%':<28} |")
    print("=" * 65)


def run_full_pipeline():
    """Hàm chạy toàn diện: nếu chưa có dataset thì tự sinh -> trích xuất -> huấn luyện"""
    base_dir = os.path.dirname(current_dir)
    training_dir = os.path.join(base_dir, "training")
    figures_dir = os.path.join(training_dir, "figures")
    cat_path = os.path.join(training_dir, "ground_truth_defects.json")
    pt_dataset_path = os.path.join(training_dir, "dataset_features.pt")
    output_model_path = os.path.join(training_dir, "defect_head.pt")
    ref_images_dir = os.path.join(base_dir, "data", "reference_images")

    # 1. Kiểm tra tập dữ liệu
    if not os.path.exists(pt_dataset_path):
        print("[AUTO] Chưa có dataset_features.pt, tiến hành tự động tạo...")
        if not os.path.exists(cat_path):
            from dataset_generator import build_synthetic_dataset
            build_synthetic_dataset(ref_images_dir, training_dir, samples_per_vase=2)
        from extract_features import extract_features_dataset
        extract_features_dataset(cat_path, pt_dataset_path)

    # 2. Huấn luyện
    train_pipeline(
        pt_dataset_path=pt_dataset_path,
        output_model_path=output_model_path,
        figures_dir=figures_dir,
        epochs=25,
        batch_size=32,
        lr=1e-3
    )


if __name__ == "__main__":
    run_full_pipeline()
