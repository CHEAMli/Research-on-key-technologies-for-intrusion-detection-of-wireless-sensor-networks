# -*- coding: utf-8 -*-
"""
TON-IoT (train_test_network.csv) — GAN+PCA 预处理 + 两阶段训练脚本（1次运行输出二分类+攻击多分类）

你要求：
  - 排除 type='normal'，保留所有攻击类型（不合并小类）
  - 一键输出两阶段结果：
      1) 二分类（Normal vs Attack）
      2) 攻击多分类（只在 Attack 上评估） + 每类样本数统计

GAN+PCA（落地版，适配 TON-IoT 高维特征）：
  - 先做 baseline 特征工程（数值 + one-hot + hashing）
  - 标准化（按训练集拟合）
  - PCA（按训练集拟合，降维到 pca_dim）
  - （可选）GAN 在“窗口级”做少数类扩增：默认只对 Stage-2（attack types）做，避免把二分类搞崩
"""

from __future__ import annotations

import os
import gc
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------
# Model import (与你现有项目结构兼容)
# -------------------------------------------------
from model_TONGAN import AnomalyDetectionModel  # type: ignore

# =========================
# 0) 配置
# =========================
@dataclass
class CFG:
    # Data
    data_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\train_test_network.csv"

    # Feature encoding
    hash_dim: int = 8
    drop_text_cols: bool = True  # dns_query/http_uri/http_user_agent 会很大，建议 drop

    # Split
    test_ratio: float = 0.20
    val_ratio_in_train: float = 0.10
    seed: int = 42

    # Window
    win: int = 64
    stride: int = 8
    window_label_mode: str = "last"  # "last" / "any_attack" / "majority"

    # GAN + PCA
    use_pca: bool = True
    pca_dim: int = 64  # 建议 >=64，避免多分类（各攻击类型）信息丢失  # 建议 32~64；太小会损失信息，太大 GAN 很难学
    pca_solver: str = "randomized"  # full/randomized/auto

    use_gan: bool = True
    gan_stage1: bool = False  # 默认 False：二分类通常已经很好，别让 GAN 破坏它
    gan_stage2: bool = True   # 默认 True：提升小类（如 mitm）
    gan_epochs: int = 20
    gan_batch: int = 256
    gan_lr: float = 2e-4
    gan_noise_dim: int = 64
    gan_min_samples: int = 50          # 小于这个数量不做 GAN
    gan_max_mult_stage1: int = 3       # Stage-1：最多扩 3 倍
    gan_max_mult_stage2: int = 10      # Stage-2：最多扩 10 倍（mitm 常常非常少）
    gan_minority_ratio_stage2: float = 0.30  # 只对少数类做 GAN（< majority*ratio）
    gan_minority_ratio_stage1: float = 0.20


    # Train
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    lr_stage1: float = 1e-3
    lr_stage2: float = 5e-4
    grad_clip: float = 1.0  # 0 表示不裁剪
    batch_size: int = 128
    weight_decay: float = 1e-4

    # Model
    model_dim: int = 128
    cross_attn_layers: int = 2
    tcn_layers: int = 2
    transformer_layers: int = 2
    nheads: int = 8
    dropout: float = 0.1

    # Stage-2
    init_from_stage1: bool = False  # 用二分类权重初始化（仅跳过 fc）
    # 每 N 个 epoch 输出一次 Stage-2 的每类指标（用于观察训练过程；太频繁会刷屏）
    stage2_report_every: int = 5
    stage2_report_digits: int = 4


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) TON Baseline 特征工程
# =========================
BOOLISH_COLS = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
    "weird_notice",
]
DASH_INT_COLS = [
    "dns_qclass", "dns_qtype", "dns_rcode",
]
LOW_CAT_COLS = [
    "proto",
    "service",
    "conn_state",
    "history",
    "dns_qclass_name",
    "dns_qtype_name",
    "ssl_version",
]
HIGH_CAT_COLS = [
    "src_ip", "dst_ip",
    "dns_query",
    "http_uri", "http_user_agent",
    "ssl_subject", "ssl_issuer",
]


def _tf_to_int(x: pd.Series) -> pd.Series:
    # True/False / "T"/"F" / 1/0
    return (
        x.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "t": 1, "f": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(np.int32)
    )


def _dashnum_to_int(x: pd.Series) -> pd.Series:
    # "-" or int-like
    x = x.astype(str).replace("-", "0")
    return pd.to_numeric(x, errors="coerce").fillna(0).astype(np.int32)


def _hash_col(col: pd.Series, dim: int, prefix: str) -> pd.DataFrame:
    """
    简单、可复现的 hashing trick：对字符串做 python hash -> bucket
    输出 dim 个数值特征（频次 0/1），避免引入 sklearn HashingVectorizer 依赖差异
    """
    s = col.fillna("missing").astype(str).values
    out = np.zeros((len(s), dim), dtype=np.uint8)
    for i, v in enumerate(s):
        h = hash(prefix + "::" + v) % dim
        out[i, h] = 1
    cols = [f"{prefix}_h{j}" for j in range(dim)]
    return pd.DataFrame(out, columns=cols)


def preprocess_ton_features(df: pd.DataFrame, cfg: CFG) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """返回：X_df、全部特征列名、连续数值列名（用于仅标准化这些列）"""
    df = df.copy()

    # 1) boolish / dash-int
    for c in BOOLISH_COLS:
        if c in df.columns:
            df[c] = _tf_to_int(df[c])
    for c in DASH_INT_COLS:
        if c in df.columns:
            df[c] = _dashnum_to_int(df[c])

    # 2) low-card one-hot
    low_cat = [c for c in LOW_CAT_COLS if c in df.columns]
    for c in low_cat:
        df[c] = df[c].fillna("missing").astype(str)
    df_low = pd.get_dummies(df[low_cat], prefix=low_cat, dtype=np.uint8) if low_cat else pd.DataFrame(index=df.index)

    # 3) high-card hashing
    high_cat = [c for c in HIGH_CAT_COLS if c in df.columns]
    if cfg.drop_text_cols:
        high_cat = [c for c in high_cat if c not in ["dns_query", "http_uri", "http_user_agent"]]
    df_hash_list = [_hash_col(df[c], cfg.hash_dim, c) for c in high_cat]
    df_hash = pd.concat(df_hash_list, axis=1) if df_hash_list else pd.DataFrame(index=df.index)

    # 4) continuous numeric
    ignore = set(low_cat + high_cat + ["label", "type"])
    cont_cols = [c for c in df.columns if c not in ignore]
    cont = df[cont_cols].copy()
    for c in cont_cols:
        cont[c] = pd.to_numeric(cont[c], errors="coerce").fillna(0).astype(np.float32)

    # concat
    X_df = pd.concat([cont, df_low, df_hash], axis=1)
    feature_cols = list(X_df.columns)
    return X_df, feature_cols, cont_cols


# =========================
# 2) 窗口与数据集
# =========================
def _window_label(y_seg: np.ndarray, normal_id: int, mode: str) -> int:
    if mode == "last":
        return int(y_seg[-1])
    if mode == "any_attack":
        return int(1 if np.any(y_seg != normal_id) else normal_id)
    # majority
    vals, cnt = np.unique(y_seg, return_counts=True)
    return int(vals[np.argmax(cnt)])


def make_windows_two_labels(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_type_str: np.ndarray,
    cfg: CFG,
    attack_type_to_idx: Dict[str, int],
    normal_token: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输出窗口：
      Xw: [N, W, D]
      yb: [N]  二分类标签（0=Normal, 1=Attack）
      yt: [N]  攻击类型标签（仅对 Attack 窗口有效；Normal/非attack 设为 -1）
    """
    xs, ybs, yts = [], [], []

    normal_id = 0
    n = len(X)
    W = cfg.win
    S = cfg.stride
    for i in range(0, n - W + 1, S):
        seg_y = y_bin[i:i + W]
        yb = _window_label(seg_y, normal_id=normal_id, mode=cfg.window_label_mode)

        # type label：取窗口最后一条的 type（与你现在的实现一致）
        t_last = str(y_type_str[i + W - 1]).lower()
        if yb == 1:
            if t_last == normal_token:
                yt = -1
            else:
                yt = int(attack_type_to_idx.get(t_last, -1))
        else:
            yt = -1

        xs.append(X[i:i + W])
        ybs.append(yb)
        yts.append(yt)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ybs, dtype=np.int64),
        np.asarray(yts, dtype=np.int64),
    )


class WindowDataset(Dataset):
    def __init__(self, Xw: np.ndarray, yw: np.ndarray):
        self.Xw = torch.from_numpy(Xw)
        self.yw = torch.from_numpy(yw)

    def __len__(self):
        return self.Xw.shape[0]

    def __getitem__(self, idx):
        return self.Xw[idx], self.yw[idx]


# =========================
# 3) GAN（窗口级扩增）
# =========================
class _GAN_G(nn.Module):
    def __init__(self, z_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class _GAN_D(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _sample_g(g: _GAN_G, n: int, z_dim: int, device: str) -> np.ndarray:
    z = torch.randn(n, z_dim, device=device)
    x = g(z).detach().cpu().numpy().astype(np.float32)
    return x


def _train_gan_one_class(
    X_real: np.ndarray,
    cfg: CFG,
    device: str,
) -> _GAN_G:
    """
    只用该类真实样本训练一个小 GAN（unconditional）。
    X_real: [N, F]
    """
    g = _GAN_G(cfg.gan_noise_dim, X_real.shape[1]).to(device)
    d = _GAN_D(X_real.shape[1]).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    # dataloader
    X_t = torch.from_numpy(X_real.astype(np.float32))
    dl = DataLoader(X_t, batch_size=min(cfg.gan_batch, len(X_t)), shuffle=True, drop_last=True)

    for epoch in range(cfg.gan_epochs):
        lossD, lossG = 0.0, 0.0
        for xb in dl:
            xb = xb.to(device)

            # D step
            opt_d.zero_grad(set_to_none=True)
            real = torch.ones((xb.size(0), 1), device=device)
            fake = torch.zeros((xb.size(0), 1), device=device)

            out_real = d(xb)
            l_real = bce(out_real, real)

            z = torch.randn(xb.size(0), cfg.gan_noise_dim, device=device)
            x_fake = g(z).detach()
            out_fake = d(x_fake)
            l_fake = bce(out_fake, fake)

            ld = l_real + l_fake
            ld.backward()
            opt_d.step()

            # G step
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(xb.size(0), cfg.gan_noise_dim, device=device)
            x_gen = g(z)
            out = d(x_gen)
            lg = bce(out, real)  # fool D
            lg.backward()
            opt_g.step()

            lossD += float(ld.item())
            lossG += float(lg.item())

        if (epoch + 1) % max(1, cfg.gan_epochs // 3) == 0:
            print(f"[GAN] epoch {epoch+1}/{cfg.gan_epochs} | lossD={lossD/len(dl):.4f} lossG={lossG/len(dl):.4f}")

    return g


def augment_windows_gan(
    Xw: np.ndarray,
    y: np.ndarray,
    cfg: CFG,
    stage: str,
    device: str,
    max_mult: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对训练窗口做 GAN 扩增到“更平衡”的分布（每类最多扩 max_mult 倍）。
    Xw: [N, W, D]
    y : [N]
    """
    assert Xw.ndim == 3
    W, D = Xw.shape[1], Xw.shape[2]
    Xf = Xw.reshape(len(Xw), W * D)

    # counts
    uniq, cnt = np.unique(y, return_counts=True)
    counts = {int(u): int(c) for u, c in zip(uniq, cnt)}
    majority = max(counts.values()) if counts else 0
    print(f"[{stage}][GAN] before: {counts} | majority={majority}")

    X_list = [Xf]
    y_list = [y]

    # 只对明显少数类做 GAN：避免把“本来已平衡的类”引入噪声，导致像你日志里 ddos/scanning/injection 这种类被压到几乎不预测。
    minority_ratio = cfg.gan_minority_ratio_stage2 if stage.lower().endswith("2") else cfg.gan_minority_ratio_stage1
    minority_cut = int(max(1, round(majority * minority_ratio)))

    for c, n_c in counts.items():
        if n_c < cfg.gan_min_samples:
            print(f"[{stage}][GAN] class {c} too small (n={n_c}), skip.")
            continue

        # 只扩少数类
        if n_c >= minority_cut:
            continue

        # 目标：最多扩 max_mult 倍，但不超过 minority_cut（也不超过 majority）
        target = min(majority, min(minority_cut, n_c * max_mult))
        need = target - n_c
        if need <= 0:
            continue

        X_real = Xf[y == c]
        print(f"[{stage}][GAN] train class {c}: n={n_c} -> target={target} (need {need})")
        g = _train_gan_one_class(X_real, cfg, device=device)

        X_new = _sample_g(g, need, cfg.gan_noise_dim, device=device)
        y_new = np.full((need,), c, dtype=np.int64)

        X_list.append(X_new)
        y_list.append(y_new)

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    # shuffle
    rng = np.random.default_rng(cfg.seed + 12345)
    idx = rng.permutation(len(X_aug))
    X_aug = X_aug[idx]
    y_aug = y_aug[idx]

    # reshape back
    Xw_aug = X_aug.reshape(len(X_aug), W, D).astype(np.float32)

    uniq2, cnt2 = np.unique(y_aug, return_counts=True)
    counts2 = {int(u): int(c) for u, c in zip(uniq2, cnt2)}
    print(f"[{stage}][GAN] after : {counts2}")

    return Xw_aug, y_aug


# =========================
# 4) 训练 / 评估
# =========================
def build_model(cfg: CFG, feature_dim: int, num_classes: int) -> nn.Module:
    return AnomalyDetectionModel(
        feature_dim=feature_dim,
        num_classes=num_classes,
        cross_attn_layers=cfg.cross_attn_layers,
        model_dim=cfg.model_dim,
        tcn_layers=cfg.tcn_layers,
        transformer_layers=cfg.transformer_layers,
        nheads=cfg.nheads,
        dropout=cfg.dropout,
    )


def train_one(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    cfg: CFG,
    num_classes: int,
    stage_name: str,
    save_path: str,
    lr: float,
    best_key: str = "f1",
    target_names: Optional[List[str]] = None,
    report_every: int = 0,
    report_digits: int = 4,
) -> str:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = -1.0
    best_path = save_path

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())

        # val
        model.eval()
        ys, ps, prob1 = [], [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(cfg.device)
                logits = model(xb)
                p = torch.softmax(logits, dim=1)
                pred = p.argmax(dim=1).cpu().numpy()
                ys.extend(yb.numpy().tolist())
                ps.extend(pred.tolist())
                if num_classes == 2:
                    prob1.extend(p[:, 1].cpu().numpy().tolist())

        if num_classes == 2:
            f1 = f1_score(ys, ps, average="binary", zero_division=0)
        else:
            f1 = f1_score(ys, ps, average="macro", zero_division=0)

        print(f"Epoch {ep:02d} | loss={tot/len(dl_tr):.4f} | VAL f1={f1:.4f}")

        # Stage-2 训练过程观察：每 report_every 个 epoch 打印一次每类指标（基于 VAL）
        if report_every and (ep % report_every == 0) and (num_classes > 2):
            tn = target_names if (target_names is not None and len(target_names) == num_classes) else None
            print(f"[{stage_name}] VAL per-class report @epoch={ep}")
            if tn is None:
                print(classification_report(ys, ps, labels=list(range(num_classes)), digits=report_digits, zero_division=0))
            else:
                print(classification_report(ys, ps, labels=list(range(num_classes)), target_names=tn, digits=report_digits, zero_division=0))

        metric = f1
        if metric > best:
            best = metric
            torch.save(model.state_dict(), best_path)

    print(f"[{stage_name}] best_val_f1={best:.4f} saved={best_path}")
    return best_path


@torch.no_grad()
def eval_binary(model: nn.Module, dl: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps, p1 = [], [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())
        p1.extend(prob[:, 1].cpu().numpy().tolist())

    ys = np.asarray(ys, dtype=np.int64)
    ps = np.asarray(ps, dtype=np.int64)
    p1 = np.asarray(p1, dtype=np.float32)

    p, r, f1, _ = precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auroc = roc_auc_score(ys, p1)

    return {
        "acc": float((ys == ps).mean()),
        "p": float(p),
        "r": float(r),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auroc": float(auroc),
    }


@torch.no_grad()
def eval_attack_multiclass(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    K: int,
    idx_to_name: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())

    ys = np.asarray(ys, dtype=np.int64)
    ps = np.asarray(ps, dtype=np.int64)

    acc = float((ys == ps).mean())
    macro_f1 = float(f1_score(ys, ps, average="macro", labels=list(range(K)), zero_division=0))
    weighted_f1 = float(f1_score(ys, ps, average="weighted", labels=list(range(K)), zero_division=0))
    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# =========================
# 5) 主流程（两阶段）
# =========================
def main():
    cfg = CFG()
    seed_all(cfg.seed)

    assert os.path.exists(cfg.data_csv), f"CSV not found: {cfg.data_csv}"
    print(f"🚀 Load CSV: {cfg.data_csv}")
    df = pd.read_csv(cfg.data_csv)
    print(f"Rows: {len(df)} | Cols: {df.shape[1]}")

    # label: 0/1; type: 'normal' + attacks
    assert "label" in df.columns and "type" in df.columns, "CSV must contain columns: label, type"
    df["type"] = df["type"].fillna("normal").astype(str).str.lower()

    # attack types set (exclude normal)
    attack_types = sorted([t for t in df["type"].unique().tolist() if t != "normal"])
    attack_type_to_idx = {t: i for i, t in enumerate(attack_types)}
    idx_to_attack = {i: t for t, i in attack_type_to_idx.items()}
    K_attack = len(attack_types)
    print(f"[Attack types] K={K_attack} (exclude 'normal')")

    # counts
    at_cnt = df[df["type"] != "normal"]["type"].value_counts()
    print("[Row counts][attack types]")
    for t in attack_types:
        print(f"  {t}: {int(at_cnt.get(t, 0))}")

    # 1) feature engineering
    X_df, feature_cols, cont_cols = preprocess_ton_features(df, cfg)
    print(f"[Features] dim={len(feature_cols)} | cont={len(cont_cols)} | hash_dim={cfg.hash_dim} | drop_text={cfg.drop_text_cols}")
    print(f"[Window] win={cfg.win} stride={cfg.stride} label_mode={cfg.window_label_mode}")

    # binary labels
    y_bin = df["label"].astype(int).to_numpy()  # 0=normal, 1=attack
    y_type = df["type"].to_numpy()

    # 2) split (stratified on binary label)
    idx = np.arange(len(X_df))
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_ratio, random_state=cfg.seed, shuffle=True, stratify=y_bin
    )
    y_train = y_bin[idx_train]
    idx_tr, idx_val = train_test_split(
        idx_train, test_size=cfg.val_ratio_in_train, random_state=cfg.seed, shuffle=True, stratify=y_train
    )
    idx_tr = np.sort(idx_tr)
    idx_val = np.sort(idx_val)
    idx_test = np.sort(idx_test)

    # 3) scale (only continuous)
    X_all = X_df.to_numpy().astype(np.float32)
    cont_idx = [feature_cols.index(c) for c in cont_cols]
    scaler = StandardScaler()
    scaler.fit(X_all[idx_tr][:, cont_idx])
    X_all[:, cont_idx] = scaler.transform(X_all[:, cont_idx]).astype(np.float32)

    # 4) PCA (fit on train only)
    if cfg.use_pca:
        pca = PCA(n_components=min(cfg.pca_dim, X_all.shape[1]), svd_solver=cfg.pca_solver, random_state=cfg.seed)
        pca.fit(X_all[idx_tr])
        X_all = pca.transform(X_all).astype(np.float32)
        ev = float(np.sum(pca.explained_variance_ratio_))
        print(f"[PCA] {X_df.shape[1]} -> {X_all.shape[1]} | explained_var={ev:.4f}")
    else:
        print("[PCA] disabled")

    # 5) windows (produce both labels)
    Xw_tr, yb_tr, yt_tr = make_windows_two_labels(X_all[idx_tr], y_bin[idx_tr], y_type[idx_tr], cfg, attack_type_to_idx)
    Xw_va, yb_va, yt_va = make_windows_two_labels(X_all[idx_val], y_bin[idx_val], y_type[idx_val], cfg, attack_type_to_idx)
    Xw_te, yb_te, yt_te = make_windows_two_labels(X_all[idx_test], y_bin[idx_test], y_type[idx_test], cfg, attack_type_to_idx)

    print(f"[Windows] train={Xw_tr.shape} val={Xw_va.shape} test={Xw_te.shape}")
    print(f"[Binary window dist][train] {dict(zip(*np.unique(yb_tr, return_counts=True)))}")

    # 6) optional GAN augmentation
    if cfg.use_gan and cfg.gan_stage1:
        Xw_tr, yb_tr = augment_windows_gan(Xw_tr, yb_tr, cfg, stage="Stage-1", device=cfg.device, max_mult=cfg.gan_max_mult_stage1)

    # ======================
    # Stage-1: Binary Detector
    # ======================
    dl_tr_bin = DataLoader(WindowDataset(Xw_tr, yb_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va_bin = DataLoader(WindowDataset(Xw_va, yb_va), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_te_bin = DataLoader(WindowDataset(Xw_te, yb_te), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model_bin = build_model(cfg, feature_dim=Xw_tr.shape[2], num_classes=2)
    print(f"[Stage-1] Model params: {sum(p.numel() for p in model_bin.parameters())}")
    best_bin = train_one(model_bin, dl_tr_bin, dl_va_bin, cfg, num_classes=2, stage_name="Stage-1", save_path="best_ton_stage1_binary.pt", lr=cfg.lr_stage1)

    model_bin.load_state_dict(torch.load(best_bin, map_location=cfg.device))
    te_bin = eval_binary(model_bin, dl_te_bin, cfg.device)
    print("\n[TEST][Stage-1 Binary]")
    print(te_bin)
    print(f"ConfusionMatrix [[TN FP],[FN TP]] = [[{te_bin['tn']} {te_bin['fp']}],[{te_bin['fn']} {te_bin['tp']}]]")

    # ======================
    # Stage-2: Attack Types (Attack windows only)
    # ======================
    mask_tr_attack = (yb_tr == 1) & (yt_tr >= 0)
    mask_va_attack = (yb_va == 1) & (yt_va >= 0)
    mask_te_attack = (yb_te == 1) & (yt_te >= 0)

    Xw_tr2, y_tr2 = Xw_tr[mask_tr_attack], yt_tr[mask_tr_attack]
    Xw_va2, y_va2 = Xw_va[mask_va_attack], yt_va[mask_va_attack]
    Xw_te2, y_te2 = Xw_te[mask_te_attack], yt_te[mask_te_attack]

    print(f"\n[Stage-2] Attack windows only: train={len(y_tr2)} val={len(y_va2)} test={len(y_te2)} | K={K_attack}")
    # per-type window counts
    print("[Stage-2 train] per-type window counts:")
    uniq, cnt = np.unique(y_tr2, return_counts=True)
    for u, c in sorted(zip(uniq, cnt), key=lambda x: x[0]):
        print(f"  {idx_to_attack[int(u)]}: {int(c)}")

    if cfg.use_gan and cfg.gan_stage2:
        Xw_tr2, y_tr2 = augment_windows_gan(Xw_tr2, y_tr2, cfg, stage="Stage-2", device=cfg.device, max_mult=cfg.gan_max_mult_stage2)

    dl_tr_type = DataLoader(WindowDataset(Xw_tr2, y_tr2), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va_type = DataLoader(WindowDataset(Xw_va2, y_va2), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_te_type = DataLoader(WindowDataset(Xw_te2, y_te2), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model_type = build_model(cfg, feature_dim=Xw_tr2.shape[2], num_classes=K_attack)

    # init from Stage-1
    if cfg.init_from_stage1:
        try:
            sd = torch.load(best_bin, map_location=cfg.device)
            # skip classifier weights if shape mismatch
            model_sd = model_type.state_dict()
            for k, v in sd.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
            model_type.load_state_dict(model_sd)
            print("[Stage-2] init_from_stage1: loaded matched weights.")
        except Exception as e:
            print(f"[Stage-2] init_from_stage1 failed: {e}")

    stage2_names = [idx_to_attack[i] for i in range(K_attack)]
    best_type = train_one(
        model_type,
        dl_tr_type,
        dl_va_type,
        cfg,
        num_classes=K_attack,
        stage_name="Stage-2",
        save_path="best_ton_stage2_attack_types.pt",
        lr=cfg.lr_stage2,
        target_names=stage2_names,
        report_every=cfg.stage2_report_every,
        report_digits=cfg.stage2_report_digits,
    )
    model_type.load_state_dict(torch.load(best_type, map_location=cfg.device))

    # attack-only report
    y_true, y_pred = [], []
    model_type.eval()
    with torch.no_grad():
        for xb, yb in dl_te_type:
            xb = xb.to(cfg.device)
            logits = model_type(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())

    print("\n[Attack-Type ClassificationReport][Attack windows only]")
    print(classification_report(y_true, y_pred, target_names=[idx_to_attack[i] for i in range(K_attack)], digits=4, zero_division=0))
    te_type = eval_attack_multiclass(model_type, dl_te_type, cfg.device, K_attack, idx_to_attack)
    print("\n[TEST][Stage-2 Attack Types | Attack windows only]")
    print(te_type)

    print(f"\n[OK] Saved: {best_bin} | {best_type}")


if __name__ == "__main__":
    main()