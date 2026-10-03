"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss đo trên 50 000 mẫu ĐẦU của X_tr (cố định cho mọi thí nghiệm)

# Cấu hình mặc định = BASELINE (M-base). lr chọn bằng val (Part 2) rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "warmup_cosine"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix_torch(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k×k (hàng = nhãn thật, cột = dự đoán), tính trên GPU bằng bincount."""
    idx = y_true.long() * k + y_pred.long()
    cm = torch.bincount(idx, minlength=k * k).reshape(k, k)
    return cm.cpu().numpy()


def per_class_scores(cm: np.ndarray):
    """Precision, recall, F1 từng lớp; bằng 0 khi mẫu số bằng 0 (giống evaluate.py)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    pred_pos = cm.sum(axis=0)
    true_pos = cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval (dropout tắt)."""
    model.eval()
    preds = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy, nhận logit thô và nhãn int64 (softmax nằm bên trong F.cross_entropy).
       "mse" : MSE giữa logit thô và one-hot của y, như nn.MSELoss: KHÔNG có hệ số 1/2,
               lấy trung bình trên MỌI phần tử (B × 7). Với reduction="sum" trả về tổng trên
               mỗi mẫu đã chia 7 (để chia N ra đúng trung bình như "mean").
    """
    logits = logits.float()                     # tính loss bằng FP32 cho ổn định (kể cả khi autocast)
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        if reduction == "sum":
            return F.mse_loss(logits, target, reduction="sum") / logits.shape[1]
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận được {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Dùng cho: train loss (tập con cố định), val, và eval cuối cùng.
    """
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = confusion_matrix_torch(y, pred)
    return {
        "loss": total_loss / len(X),
        "acc": (pred == y).float().mean().item(),
        "macro_f1": macro_f1_from_confusion(cm),
    }


def _autocast_ctx(precision: str, device_type: str):
    """Ngữ cảnh autocast: chỉ bọc forward + loss. FP32 -> không làm gì."""
    if precision == "fp32":
        return torch.autocast(device_type=device_type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Trả về dict:
        {"cfg", "history", "summary", "best_state"}
    history theo epoch: epoch, train_loss, val_loss, val_acc, val_macro_f1, grad_norm (trung bình,
        TRƯỚC clip), grad_norm_max, clip_frac (tỉ lệ bước bị cắt), lr, epoch_time_s
    summary: step0_loss, best_val_loss, best_epoch, final_train_loss, final_val_loss, val_acc,
        val_macro_f1 (hai cái này lấy ở best_epoch), time_per_epoch_s, peak_mem_MB, diverged, ...
    best_state: state_dict (trên CPU) của epoch có val_loss thấp nhất.
    KHÔNG dùng X_eval ở đây: chọn epoch/cấu hình chỉ bằng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] đang là None: hãy chọn lr bằng val rồi truyền vào")
    if cfg["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {cfg['precision']!r}")

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    device_type = device.type
    use_cuda = device_type == "cuda"
    if cfg["precision"] == "fp16" and not use_cuda:
        raise RuntimeError("FP16 + GradScaler cần GPU CUDA")

    # ---- 0. Seed, model, optimizer --------------------------------------------------------
    set_seed(cfg["seed"])
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[cfg["hidden"]], (n_params, cfg["hidden"])

    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
                                betas=cfg.get("betas", (0.9, 0.999)), eps=cfg.get("eps", 1e-8))
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps,
                                **cfg.get("scheduler_kwargs", {}))
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])   # thứ tự xáo lô theo seed

    X_trsub, y_trsub = X_tr[:TRAIN_EVAL_SUBSET], y_tr[:TRAIN_EVAL_SUBSET]
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. Loss bước 0 (trước bước cập nhật đầu tiên) ----------------------------------------
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    if verbose:
        print(f"[{cfg['exp_id']}] {n_params:,} tham số | loss bước 0 (val) = {step0['loss']:.4f}"
              f" | {steps_per_epoch} bước/epoch")

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "lr", "epoch_time_s")}
    best_val_loss, best_epoch, best_state = float("inf"), None, None
    diverged, diverged_at = False, None
    global_step = 0

    # ---- 2. Vòng huấn luyện ----------------------------------------------------------------
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gnorms, n_clipped = [], 0

        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with _autocast_ctx(cfg["precision"], device_type):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if not torch.isfinite(loss):
                diverged, diverged_at = True, (epoch, global_step)
                break

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)          # đưa grad về thang thật TRƯỚC khi đo/cắt
            else:
                loss.backward()

            gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt

            if scaler is not None:
                scaler.step(optimizer)              # tự bỏ qua bước nếu grad có inf/NaN (tràn FP16)
                scaler.update()
            else:
                if not math.isfinite(gn):
                    diverged, diverged_at = True, (epoch, global_step)
                    break
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if math.isfinite(gn):                   # bước tràn FP16 bị scaler bỏ qua: không tính
                gnorms.append(gn)
                if cfg["clip_norm"] is not None and gn > cfg["clip_norm"]:
                    n_clipped += 1

        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if diverged:
            if verbose:
                print(f"  !! DIVERGED ở epoch {epoch}, bước {global_step}: loss thành NaN/inf, dừng sớm")
            break

        # ---- cuối epoch: đánh giá ở chế độ eval ----
        tr = evaluate(model, X_trsub, y_trsub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            diverged, diverged_at = True, (epoch, global_step)
            if verbose:
                print(f"  !! DIVERGED sau epoch {epoch}: loss đánh giá là NaN/inf")
            break

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(gnorms)) if gnorms else float("nan"))
        hist["grad_norm_max"].append(float(np.max(gnorms)) if gnorms else float("nan"))
        hist["clip_frac"].append(n_clipped / max(1, len(gnorms)))
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        hist["epoch_time_s"].append(epoch_time)

        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            mark = " *" if best_epoch == epoch else ""
            print(f"  ep {epoch:2d} | train {tr['loss']:.4f} | val {va['loss']:.4f} | "
                  f"acc {va['acc']:.4f} | mF1 {va['macro_f1']:.4f} | "
                  f"gn {hist['grad_norm'][-1]:.3f} | {epoch_time:.1f}s{mark}")

    # ---- 3. Tóm tắt ở best_epoch ----------------------------------------------------------
    if best_epoch is not None:
        bi = hist["epoch"].index(best_epoch)
        val_acc, val_f1 = hist["val_acc"][bi], hist["val_macro_f1"][bi]
    else:
        val_acc = val_f1 = float("nan")
    summary = {
        "n_params": n_params,
        "step0_loss": step0["loss"],
        "best_val_loss": best_val_loss if best_epoch is not None else float("nan"),
        "best_epoch": best_epoch,
        "final_train_loss": hist["train_loss"][-1] if hist["train_loss"] else float("nan"),
        "final_val_loss": hist["val_loss"][-1] if hist["val_loss"] else float("nan"),
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "final_val_acc": hist["val_acc"][-1] if hist["val_acc"] else float("nan"),
        "final_val_macro_f1": hist["val_macro_f1"][-1] if hist["val_macro_f1"] else float("nan"),
        "grad_norm_mean": float(np.nanmean(hist["grad_norm"])) if hist["grad_norm"] else float("nan"),
        "clip_frac": float(np.mean(hist["clip_frac"])) if hist["clip_frac"] else 0.0,
        "time_per_epoch_s": float(np.mean(hist["epoch_time_s"])) if hist["epoch_time_s"] else float("nan"),
        "peak_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else float("nan"),
        "total_steps": global_step,
        "diverged": diverged,
        "diverged_at": diverged_at,
        "device": torch.cuda.get_device_name(0) if use_cuda else str(device),
    }
    if verbose:
        print(f"  => best epoch {best_epoch}: val loss {summary['best_val_loss']:.4f}, "
              f"acc {val_acc:.4f}, macro-F1 {val_f1:.4f} | {summary['time_per_epoch_s']:.1f}s/epoch")

    out_cfg = copy.deepcopy(cfg)
    out_cfg["hidden"] = list(out_cfg["hidden"])
    return {"cfg": out_cfg, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV tiêu đề `row_id,pred`, đủ mọi dòng eval."""
    import pandas as pd

    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds), (len(row_id), len(preds))
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải trong 0..6"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)
    print(f"Đã ghi {len(preds):,} dự đoán -> {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions,
    rồi (nếu có repo_root) chạy scripts/evaluate.py và trả về nội dung eval_result.json.
    """
    import json

    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg.get("dropout", 0.0), init="default")
    model.load_state_dict(result["best_state"])
    model = model.to(data["X_eval"].device)
    preds = predict(model, data["X_eval"])                # fp32, eval mode (dropout tắt)
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)

    if repo_root is None:
        return None
    pred_abs = os.path.abspath(pred_path)
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", pred_abs]
    out_abs = None
    if out_json is not None:
        out_abs = os.path.abspath(out_json)
        cmd += ["--out", out_abs]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("scripts/evaluate.py báo lỗi, xem thông báo ở trên")
    if out_abs is not None:
        with open(out_abs, encoding="utf-8") as f:
            return json.load(f)
    return None