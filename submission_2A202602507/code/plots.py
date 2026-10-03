"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng so sánh (figures/compare_<nhóm>.png).

Notebook chạy trong code/ nên lưu vào "../figures/".
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 100

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)",
    "val_loss": "val loss",
    "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1",
    "grad_norm": "grad norm trung bình (trước clip)",
    "grad_norm_max": "grad norm lớn nhất (trước clip)",
    "epoch_time_s": "thời gian / epoch (s)",
}


def _cfg_text(cfg: dict) -> str:
    """Chuỗi cấu hình ngắn gọn để đặt trong tiêu đề."""
    hidden = "-".join(str(h) for h in cfg.get("hidden", ()))
    parts = [
        f"{cfg.get('optimizer')} lr={cfg.get('lr'):g}",
        f"loss={cfg.get('loss')}",
        f"batch={cfg.get('batch')}",
        f"hidden={hidden}",
        f"init={cfg.get('init')}",
        f"drop={cfg.get('dropout')}",
        f"clip={cfg.get('clip_norm')}",
        f"{cfg.get('precision')}",
        f"wd={cfg.get('weight_decay')}",
        f"seed={cfg.get('seed')}",
    ]
    if cfg.get("scheduler") not in (None, "none"):
        parts.append(f"sched={cfg.get('scheduler')}")
    return ", ".join(parts)


def plot_run(result: dict, path: str, show: bool = False) -> None:
    """Một thí nghiệm -> một PNG gồm 3 ô:
         (1) train_loss và val_loss theo epoch
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm (trung bình và lớn nhất mỗi epoch, đo TRƯỚC khi clip)
    Đường đứt nét đánh dấu best_epoch (val_loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    best = s.get("best_epoch")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode)")
    ax.plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    if s.get("step0_loss") is not None:
        ax.axhline(s["step0_loss"], color="gray", ls=":", lw=1, label=f"loss bước 0 = {s['step0_loss']:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel(f"loss ({cfg.get('loss')})")
    ax.set_title("(1) Loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán lớp đa số (acc 0.4876)")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm")
    ax.set_title("(2) Val accuracy / macro-F1")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="trung bình")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.7, label="lớn nhất")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls="--", lw=1, label=f"ngưỡng clip c = {cfg['clip_norm']}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("||g||₂ (thang log)")
    ax.set_title("(3) Grad norm (trước clip)")

    for ax in axes:
        if best is not None:
            ax.axvline(best, color="green", ls="--", lw=1, alpha=0.6, label=f"best epoch = {best}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    status = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg.get('exp_id')}{status} — {_cfg_text(cfg)}\n"
                 f"best epoch {best}: val acc = {s.get('val_acc', float('nan')):.4f}, "
                 f"val macro-F1 = {s.get('val_macro_f1', float('nan')):.4f}",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "",
                 show: bool = False, logy: bool | None = None) -> None:
    """Vẽ chồng một hoặc nhiều chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường (chú thích = exp_id).

    metric: tên một chỉ số ("val_loss", "val_macro_f1", "grad_norm", ...) hoặc list nhiều chỉ số
            (mỗi chỉ số một ô). Dùng cho figures/compare_<nhóm>.png.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            h = r["history"]
            if m not in h:
                continue
            ax.plot(h["epoch"], h[m], "o-", ms=3, label=r["cfg"]["exp_id"])
        use_log = logy if logy is not None else m.startswith("grad_norm")
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)