"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối: không dùng để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10    # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")

    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    # Kiểm tra đúng quy ước, để lỗi dữ liệu lộ ra ngay ở đây chứ không phải lúc huấn luyện
    for X, y in [(X_train_full, y_train_full), (X_eval, y_eval)]:
        assert X.ndim == 2 and X.shape[1] == N_FEATURES, X.shape
        assert X.dtype == np.float32, X.dtype
        assert y.dtype == np.int64, y.dtype
        assert len(X) == len(y)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1
    assert len(eval_row_id) == len(X_eval)

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Mọi thí nghiệm dùng CÙNG seed và val_fraction để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split

    # stratify=y giữ tỉ lệ 7 lớp ở train và val như nhau (lớp 3 chỉ ~0,5%)
    # Chú ý thứ tự trả về của sklearn là X_tr, X_val, y_tr, y_val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (10,), std (10,)
    Không tính trên val/eval: làm vậy là để thống kê của dữ liệu kiểm tra "rò rỉ" vào
    bước tiền xử lý, khiến điểm val/eval lạc quan hơn thực tế.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)   # tính bằng float64 cho chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về BẢN SAO của X: 10 cột đầu được (x - mean) / std, 44 cột nhị phân giữ nguyên."""
    X2 = X.copy()                                   # không sửa X gốc
    safe_std = np.where(std == 0, 1.0, std)         # tránh chia cho 0 nếu có cột hằng
    X2[:, :N_NUMERIC] = (X2[:, :N_NUMERIC] - mean) / safe_std
    return X2.astype(np.float32, copy=False)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict:
        tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval
        numpy:              eval_row_id, mean, std
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)              # CHỈ dùng X_tr
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)    # cùng mean/std của train
    X_eval = apply_standardizer(X_eval, mean, std)

    def to_x(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    def to_y(a):
        return torch.as_tensor(a, dtype=torch.int64, device=device)

    data = {
        "X_tr": to_x(X_tr), "y_tr": to_y(y_tr),
        "X_val": to_x(X_val), "y_val": to_y(y_val),
        "X_eval": to_x(X_eval), "y_eval": to_y(y_eval),
        "eval_row_id": eval_row_id, "mean": mean, "std": std,
    }

    if verbose:
        print(f"train: {len(y_tr):,}  val: {len(y_val):,}  eval: {len(y_eval):,}")
        print("tỉ lệ lớp (%)")
        print("lớp   train     val    eval")
        for c in range(N_CLASSES):
            print(f"{c:>3d}  {100*np.mean(y_tr == c):6.3f}  {100*np.mean(y_val == c):6.3f}"
                  f"  {100*np.mean(y_eval == c):6.3f}")
        maj = np.bincount(y_tr, minlength=N_CLASSES).argmax()
        print(f"đoán luôn lớp đa số (lớp {maj}) -> accuracy trên val = {np.mean(y_val == maj):.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None,
                    shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Lô cuối có thể nhỏ hơn batch_size; ở đây GIỮ lại lô đó (không bỏ dữ liệu).
    `generator` phải nằm trên cùng thiết bị với X (vd torch.Generator(device="cuda")).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]