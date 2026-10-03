"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (README mục 5).

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr*wd*w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "none", "cosine", "warmup_cosine")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam là L2 trộn vào gradient (bị chia cho sqrt(v_hat)),
    còn của AdamW là suy giảm tách riêng, không phụ thuộc độ lớn gradient.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận được {name!r}")
    if lr is None or lr <= 0:
        raise ValueError(f"lr phải là số dương, nhận được {lr!r}")
    betas = tuple(betas)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch lr, gọi scheduler.step() SAU MỖI BƯỚC cập nhật (không phải mỗi epoch).

    name:
        None / "none"   : không dùng (trả về None)
        "cosine"        : lr giảm theo cosine từ lr ban đầu về ~0 trong total_steps bước
        "warmup_cosine" : tăng tuyến tính trong `warmup_steps` bước đầu (mặc định 5% tổng),
                          sau đó giảm theo cosine. Dùng cho thí nghiệm "lô ×k thì lr ×k, kèm khởi động".
    Nếu dùng scheduler ở thí nghiệm nào, ghi vào cột notes của bảng.
    """
    if name in (None, "none"):
        return None
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải thuộc {SCHEDULERS}, nhận được {name!r}")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    warmup = int(kwargs.get("warmup_steps", max(1, int(0.05 * total_steps))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 TOÀN CỤC và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None: chỉ đo chuẩn, không cắt (dùng max_norm = inf).
    Công thức khi cắt: g <- g * min(1, max_norm / ||g||).
    Với FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này,
    nếu không chuẩn đo được là chuẩn của gradient đã bị nhân hệ số scale.
    """
    params = [p for p in params if p.grad is not None]
    limit = float("inf") if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, limit)
    return float(total_norm)