# -*- coding: utf-8 -*-
"""
TON-IoT (train_test_network.csv) — Baseline 预处理 + 两阶段训练脚本（1次运行输出二分类+攻击多分类）

目标（与你在 WSN-DS 的写法一致）：
  Stage-1: Binary Detector  (Normal vs Attack)  —— 使用 df['label']
  Stage-2: Attack Classifier (Attack types)     —— 排除 type='normal'，保留全部攻击类型

输出：
  1) 二分类：P/R/F1/AUROC + 2×2 混淆矩阵
  2) 攻击多分类：只在 ground-truth Attack 窗口上评估，给 macro/weighted-F1 + 每类窗口数统计

说明：
  - 仍使用 baseline 特征工程：数值列 scaler；低基数 one-hot；高基数 hashing（可控维度）
  - 仍按“原始行序”滑窗构造序列（win=64），默认窗口标签使用 last-step（更接近流式检测）
  - Stage-2 默认用 Stage-1 的特征抽取权重初始化（仅跳过 fc 分类头），更稳更快

使用：
  1) 修改 CFG.data_csv 为你的 train_test_network.csv 路径
  2) python trainTON_two_stage.py
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# 兼容：你可能用的是 modelTON_clean.py（我之前给你的“清理版”），也可能是原始 modelTON.py
try:
    from modelTON_clean import AnomalyDetectionModel  # type: ignore
except Exception:
    from modelTON import AnomalyDetectionModel  # type: ignore


# =========================
# 0) 配置
# =========================
@dataclass
class CFG:
    # Data
    data_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\train_test_network.csv"

    # Feature encoding
    hash_dim: int = 8
    drop_text_cols: bool = True

    # Windowing
    win: int = 64
    stride: int = 8
    window_label_mode: str = "last"  # "majority" / "last" / "any_attack"

    # Split
    test_ratio: float = 0.2
    val_ratio_in_train: float = 0.1

    # Train
    seed: int = 42
    batch_size: int = 256
    epochs_bin: int = 20
    epochs_type: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model_dim: int = 128
    cross_attn_layers: int = 2
    tcn_layers: int = 2
    transformer_layers: int = 2
    nheads: int = 8
    dropout: float = 0.1

    # Stage-2
    init_from_stage1: bool = True  # 用二分类权重初始化（仅跳过 fc）


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 1) TON Baseline 预处理（只做特征，不掺杂标签）
# =========================
BOOLISH_COLS = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
    "weird_notice",
]

DASH_INT_COLS = [
    "http_trans_depth", "weird_addl",
]

LOW_CAT_COLS = [
    "proto", "service", "conn_state",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "ssl_version", "ssl_cipher",
    "http_method", "http_version",
    "weird_name",
]

HIGH_CAT_COLS = [
    "src_ip", "dst_ip",
    "dns_query",
    "http_uri",
    "http_user_agent",
    "ssl_subject",
    "ssl_issuer",
    "http_orig_mime_types",
    "http_resp_mime_types",
]


def _tf_to_int(s: pd.Series) -> pd.Series:
    return (
        s.fillna("-")
        .astype(str)
        .map({"T": 1, "F": 0, "-": 0})
        .fillna(0)
        .astype(np.int8)
    )


def _dashnum_to_int(s: pd.Series) -> pd.Series:
    return (
        s.fillna("-")
        .astype(str)
        .replace("-", "0")
        .astype(np.int32)
    )


def _hash_col(series: pd.Series, n_features: int, prefix: str) -> pd.DataFrame:
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    tokens = [[f"{prefix}={v}"] for v in series.fillna("missing").astype(str).tolist()]
    mat = hasher.transform(tokens).toarray().astype(np.float32)
    return pd.DataFrame(mat, columns=[f"{prefix}_h{i}" for i in range(n_features)])


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
    for c in cont.columns:
        cont[c] = pd.to_numeric(cont[c], errors="coerce")
    cont = cont.fillna(0.0).astype(np.float32)

    X_df = pd.concat([cont, df_low, df_hash], axis=1)
    return X_df, list(X_df.columns), list(cont.columns)


# =========================
# 2) 滑窗（同时产出 binary label + attack type label）
# =========================
def window_label_from_seg(y_seg: np.ndarray, mode: str, normal_id: int = 0) -> int:
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
    返回：
      Xw: [Nw, W, D]
      yb_w: [Nw]  (0/1)
      yt_w: [Nw]  (攻击类型 idx；非攻击窗口为 -1)
    """
    N, D = X.shape
    if N < cfg.win:
        raise ValueError(f"Too few rows ({N}) for win={cfg.win}")

    xs, ybs, yts = [], [], []
    for i in range(0, N - cfg.win + 1, cfg.stride):
        seg_bin = y_bin[i:i + cfg.win]
        yb = window_label_from_seg(seg_bin, cfg.window_label_mode, normal_id=0)
        # type 用 last-step，更符合“当前时刻属于哪类攻击”
        t_last = str(y_type_str[i + cfg.win - 1])
        if yb == 1:
            # 只在攻击窗口上保留 type label
            if t_last.lower() == normal_token:
                yt = -1
            else:
                yt = int(attack_type_to_idx.get(t_last, -1))
        else:
            yt = -1

        xs.append(X[i:i + cfg.win])
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
# 3) 评估
# =========================
@torch.no_grad()
def eval_binary(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob >= 0.5).long()
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pr = np.concatenate(probs)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    out = {
        "acc": float(accuracy_score(y, p)),
        "p": float(precision_score(y, p, zero_division=0)),
        "r": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    try:
        out["auroc"] = float(roc_auc_score(y, pr))
    except Exception:
        out["auroc"] = float("nan")
    return out


@torch.no_grad()
def eval_attack_multiclass(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    K: int,
    target_names: List[str],
) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        ys.append(yb.numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    out = {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(K)), zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", labels=list(range(K)), zero_division=0)),
    }
    try:
        print("\n[Attack-Type ClassificationReport][Attack windows only]")
        print(classification_report(y, p, labels=list(range(K)), target_names=target_names, zero_division=0, digits=4))
    except Exception:
        pass
    return out


def balanced_weights(y: np.ndarray, K: int) -> np.ndarray:
    """balanced 权重：w_c = N/(K*count_c)，缺失类 weight=1"""
    y = y.astype(np.int64)
    counts = np.bincount(y, minlength=K).astype(np.float64)
    N = counts.sum()
    w = np.ones(K, dtype=np.float32)
    for c in range(K):
        if counts[c] > 0:
            w[c] = float(N / (K * counts[c]))
    return w


# =========================
# 4) 主流程（两阶段）
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
        max_len=cfg.win,
    )


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CFG,
    epochs: int,
    criterion: nn.Module,
    eval_fn,
    best_path: str,
    best_key: str,
):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        avg_loss = total / max(1, len(train_loader))
        val = eval_fn(model, val_loader)
        score = float(val[best_key])
        print(f"Epoch {ep:02d} | loss={avg_loss:.4f} | VAL {best_key}={score:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=cfg.device))


def main():
    cfg = CFG()
    seed_all(cfg.seed)
    assert cfg.window_label_mode in ("majority", "last", "any_attack")

    print("🚀 Load CSV:", cfg.data_csv)
    df = pd.read_csv(cfg.data_csv, low_memory=False)
    print("Rows:", len(df), "| Cols:", df.shape[1])

    # labels
    y_bin = df["label"].astype(int).to_numpy()
    type_raw = df["type"].fillna("missing").astype(str).to_numpy()
    # attack types: exclude 'normal'
    is_normal_type = np.char.lower(type_raw.astype(str)) == "normal"
    attack_type_set = sorted(set(type_raw[~is_normal_type].tolist()))
    attack_type_to_idx = {t: i for i, t in enumerate(attack_type_set)}
    print(f"[Attack types] K={len(attack_type_set)} (exclude 'normal')")
    # row-level counts (for论文/复现实验更好写)
    if len(attack_type_set) > 0:
        vc = pd.Series(type_raw[~is_normal_type]).value_counts()
        print("[Row counts][attack types]")
        for name, c in vc.items():
            print(f"  {name}: {int(c)}")

    # 1) features
    X_df, feature_cols, cont_cols = preprocess_ton_features(df, cfg)
    print(f"[Features] dim={len(feature_cols)} | cont={len(cont_cols)} | hash_dim={cfg.hash_dim} | drop_text={cfg.drop_text_cols}")
    print(f"[Window] win={cfg.win} stride={cfg.stride} label_mode={cfg.window_label_mode}")

    # 2) split on binary label (stratified)
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

    # 4) windows (produce both labels)
    Xw_tr, yb_tr, yt_tr = make_windows_two_labels(X_all[idx_tr], y_bin[idx_tr], type_raw[idx_tr], cfg, attack_type_to_idx)
    Xw_val, yb_val, yt_val = make_windows_two_labels(X_all[idx_val], y_bin[idx_val], type_raw[idx_val], cfg, attack_type_to_idx)
    Xw_te, yb_te, yt_te = make_windows_two_labels(X_all[idx_test], y_bin[idx_test], type_raw[idx_test], cfg, attack_type_to_idx)
    print(f"[Windows] train={Xw_tr.shape} val={Xw_val.shape} test={Xw_te.shape}")
    print("[Binary window dist][train]", {int(k): int(v) for k, v in zip(*np.unique(yb_tr, return_counts=True))})

    # =========================
    # Stage-1: Binary Detector
    # =========================
    feature_dim = Xw_tr.shape[2]
    model_bin = build_model(cfg, feature_dim, num_classes=2).to(cfg.device)
    print("[Stage-1] Model params:", sum(p.numel() for p in model_bin.parameters()))

    # weights + loss
    w_bin = balanced_weights(yb_tr, K=2)
    crit_bin = nn.CrossEntropyLoss(weight=torch.tensor(w_bin, dtype=torch.float32, device=cfg.device))

    dl_tr_bin = DataLoader(WindowDataset(Xw_tr, yb_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val_bin = DataLoader(WindowDataset(Xw_val, yb_val), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_te_bin = DataLoader(WindowDataset(Xw_te, yb_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    best_bin = "best_ton_stage1_binary.pt"
    train_loop(
        model_bin,
        dl_tr_bin,
        dl_val_bin,
        cfg,
        epochs=cfg.epochs_bin,
        criterion=crit_bin,
        eval_fn=lambda m, l: eval_binary(m, l, cfg.device),
        best_path=best_bin,
        best_key="f1",
    )

    te_bin = eval_binary(model_bin, dl_te_bin, cfg.device)
    print("\n[TEST][Stage-1 Binary]")
    print(te_bin)
    print(f"ConfusionMatrix [[TN FP],[FN TP]] = [[{te_bin['tn']} {te_bin['fp']}],[{te_bin['fn']} {te_bin['tp']}]]")

    # =========================
    # Stage-2: Attack Type Classifier (Attack windows only)
    # =========================
    # filter windows where ground-truth is attack and type label is valid
    def filter_attack(Xw, yt, yb):
        m = (yb == 1) & (yt >= 0)
        return Xw[m], yt[m]

    Xw_tr_a, yt_tr_a = filter_attack(Xw_tr, yt_tr, yb_tr)
    Xw_val_a, yt_val_a = filter_attack(Xw_val, yt_val, yb_val)
    Xw_te_a, yt_te_a = filter_attack(Xw_te, yt_te, yb_te)

    K_attack = len(attack_type_set)
    print(f"\n[Stage-2] Attack windows only: train={len(yt_tr_a)} val={len(yt_val_a)} test={len(yt_te_a)} | K={K_attack}")

    # print per-class window counts
    def show_counts(title: str, y: np.ndarray):
        cnt = np.bincount(y, minlength=K_attack)
        pairs = [(attack_type_set[i], int(cnt[i])) for i in range(K_attack)]
        print(f"[{title}] per-type window counts:")
        for name, c in pairs:
            if c > 0:
                print(f"  {name}: {c}")
        # 也输出总数
        print(f"  TOTAL: {int(cnt.sum())}")

    show_counts("Stage-2 train", yt_tr_a)
    show_counts("Stage-2 val", yt_val_a)
    show_counts("Stage-2 test", yt_te_a)

    if len(yt_tr_a) == 0 or K_attack == 0:
        print("[WARN] No attack windows or no attack types found. Stage-2 skipped.")
        return

    model_type = build_model(cfg, feature_dim, num_classes=K_attack).to(cfg.device)

    # init from stage-1 (skip fc)
    if cfg.init_from_stage1:
        sd = torch.load(best_bin, map_location="cpu")
        # remove classifier head
        for k in list(sd.keys()):
            if k.startswith("fc."):
                sd.pop(k)
        missing, unexpected = model_type.load_state_dict(sd, strict=False)
        print("[Stage-2] init_from_stage1=True | missing:", len(missing), "unexpected:", len(unexpected))

    w_type = balanced_weights(yt_tr_a, K=K_attack)
    crit_type = nn.CrossEntropyLoss(weight=torch.tensor(w_type, dtype=torch.float32, device=cfg.device))

    dl_tr_type = DataLoader(WindowDataset(Xw_tr_a, yt_tr_a), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val_type = DataLoader(WindowDataset(Xw_val_a, yt_val_a), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    dl_te_type = DataLoader(WindowDataset(Xw_te_a, yt_te_a), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    best_type = "best_ton_stage2_attack_types.pt"
    train_loop(
        model_type,
        dl_tr_type,
        dl_val_type,
        cfg,
        epochs=cfg.epochs_type,
        criterion=crit_type,
        eval_fn=lambda m, l: eval_attack_multiclass(m, l, cfg.device, K_attack, attack_type_set),
        best_path=best_type,
        best_key="macro_f1",
    )

    te_type = eval_attack_multiclass(model_type, dl_te_type, cfg.device, K_attack, attack_type_set)
    print("\n[TEST][Stage-2 Attack Types | Attack windows only]")
    print(te_type)

    print(f"\n[OK] Saved: {best_bin} | {best_type}")


if __name__ == "__main__":
    main()
