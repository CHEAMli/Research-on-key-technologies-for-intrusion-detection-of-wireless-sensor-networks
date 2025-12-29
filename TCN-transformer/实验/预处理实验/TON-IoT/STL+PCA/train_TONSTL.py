# -*- coding: utf-8 -*-
"""
TON-IoT 两阶段实验（STL+PCA 预处理版本）—— 严格“调用你的模型文件”
- Stage-1: 二分类 Normal vs Attack（使用 label 列）
- Stage-2: 攻击多分类（仅在 Attack 子集上评估，排除 type=normal，保留所有攻击类型，不合并小类）

你现在的约束：
- 数据集：train_test_network.csv（单文件，脚本内做 split）
- 预处理：STL + PCA（先做数值编码/标准化，再做 STL 去趋势/残差，再 PCA 降维）
- 模型：从 model_TONSTL.py 导入 AnomalyDetectionModel（不在训练脚本里定义模型）

运行：
python train_TONSTL_two_stage_stl_pca.py

依赖：
pip install pandas numpy scikit-learn torch
可选（更“真”STL）：pip install statsmodels
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    classification_report,
)

# ============================================================
# 0) 你的模型导入（关键：必须从你的 model_TONSTL.py 来）
# ============================================================
from model_TONSTL import AnomalyDetectionModel  # type: ignore



# ============================================================
# 1) 配置
# ============================================================
@dataclass
class CFG:
    # Data
    data_csv: str = r"C:\Users\yeqing\PycharmProjects\pythonProject\Train_Test_datasets\Train_Test_Network_dataset\train_test_network.csv"

    # Split
    test_ratio: float = 0.20
    val_ratio_in_train: float = 0.10
    seed: int = 42

    # Windowing
    win: int = 64
    stride: int = 1

    # Feature encoding
    drop_text_cols: bool = True  # dns_query/http_uri/http_user_agent 会很大，建议 drop
    hash_dim: int = 8            # 对 (src_ip,dst_ip,proto,service,conn_state,...) 做 hash one-hot

    # STL + PCA
    pca_dim: int = 64
    stl_period: int = 64         # “周期/季节长度”，通常用窗口长度 win
    stl_robust: bool = True
    # 如果你环境没装 statsmodels，会自动回退到 rolling-trend（速度快但不是真 STL）
    # 想强制 rolling：把 stl_force_rolling=True
    stl_force_rolling: bool = False
    rolling_win: int = 64        # rolling 回退时的趋势窗口

    # Train
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Model
    model_dim: int = 128
    tcn_layers: int = 2
    transformer_layers: int = 2
    nheads: int = 8
    dropout: float = 0.1
    max_len: int = 512


# ============================================================
# 2) 工具函数
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_sorted_unique(values: List[str]) -> List[str]:
    # 保证 attack type 顺序稳定（避免“乱序”）
    return sorted(pd.unique(pd.Series(values)).tolist())


def hash_to_onehot(series: pd.Series, dim: int) -> np.ndarray:
    """把一个字符串列 hash 成 dim 维 one-hot（稳定、可复现）"""
    # pandas 的 hash 可能跨进程不稳定，这里用 Python 的 md5 生成稳定 hash
    import hashlib

    def h(x: str) -> int:
        s = str(x)
        digest = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:8], 16) % dim

    idx = series.astype(str).map(h).to_numpy()
    out = np.zeros((len(idx), dim), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def build_feature_matrix(df: pd.DataFrame, cfg: CFG) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    把原始 dataframe -> 数值特征矩阵 X (float32)
    返回：
      X: [N, D]
      meta: 记录使用的列等（便于复现实验）
    """
    df = df.copy()

    # 目标列
    assert "label" in df.columns, "TON-IoT 文件需要有 label 列"
    assert "type" in df.columns, "TON-IoT 文件需要有 type 列"

    # 1) 丢弃一些超长文本列（可选）
    if cfg.drop_text_cols:
        for c in ["dns_query", "http_uri", "http_user_agent"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    # 2) 分离 label/type
    y_bin = df["label"].astype(int).to_numpy()  # 0 normal, 1 attack
    t_type = df["type"].astype(str).to_numpy()

    # 3) 构建数值 + hash one-hot
    drop_cols = ["label", "type"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 数值列
    num_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    X_num = X_df[num_cols].fillna(0).astype(np.float32).to_numpy()

    # 类别/文本列做 hash one-hot
    cat_cols = [c for c in X_df.columns if c not in num_cols]
    X_cats = []
    for c in cat_cols:
        X_cats.append(hash_to_onehot(X_df[c].fillna("-"), cfg.hash_dim))
    X_cat = np.concatenate(X_cats, axis=1).astype(np.float32) if X_cats else np.zeros((len(df), 0), np.float32)

    X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "hash_dim": [str(cfg.hash_dim)],
    }
    return X, {"meta": meta, "y_bin": y_bin.tolist(), "type": t_type.tolist()}


def stl_residual_matrix(X: np.ndarray, cfg: CFG, fit_on: str = "train") -> np.ndarray:
    """
    对每一维特征做 STL 分解并取 resid（如果 statsmodels 不可用则回退 rolling trend）。
    输入 X: [N, D]
    输出 X_resid: [N, D]
    """
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape

    use_rolling = cfg.stl_force_rolling
    STL = None
    if not use_rolling:
        try:
            from statsmodels.tsa.seasonal import STL as _STL  # type: ignore
            STL = _STL
        except Exception:
            use_rolling = True

    if use_rolling:
        # 回退方案：rolling mean 当作 trend，resid = x - trend（速度快但不是真 STL）
        print(f"[STL] statsmodels 不可用或强制 rolling，使用 rolling-trend 回退（fit_on={fit_on}）")
        Xr = np.empty_like(X)
        w = max(3, int(cfg.rolling_win))
        # 用 pandas rolling 处理边界更稳
        for j in range(D):
            s = pd.Series(X[:, j])
            trend = s.rolling(window=w, center=True, min_periods=max(1, w // 2)).mean()
            trend = trend.bfill().ffill().to_numpy(dtype=np.float32)
            Xr[:, j] = X[:, j] - trend
        return Xr

    # 真 STL
    print(f"[STL] 使用 statsmodels STL(period={cfg.stl_period}, robust={cfg.stl_robust})（fit_on={fit_on}）")
    Xr = np.empty_like(X)
    period = int(cfg.stl_period)
    for j in range(D):
        series = X[:, j].astype(np.float64)
        res = STL(series, period=period, robust=cfg.stl_robust).fit()
        Xr[:, j] = res.resid.astype(np.float32)
    return Xr



def apply_stl_subset(X: np.ndarray, idx: list[int], cfg: CFG, fit_on: str) -> np.ndarray:
    """Apply STL surrogate only on selected feature indices."""
    if not idx:
        return X
    X2 = X.copy()
    X2[:, idx] = stl_residual_matrix(X2[:, idx], cfg, fit_on=fit_on)
    return X2


def make_windows(X: np.ndarray, y: np.ndarray, win: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    以行序列为时间轴做滑窗：X [N,D] -> Xw [M,win,D], yw [M]
    标签取窗口最后一个样本的标签（和你 WSN 的做法一致）
    """
    N, D = X.shape
    idx_end = np.arange(win - 1, N, stride)
    M = len(idx_end)
    Xw = np.zeros((M, win, D), dtype=np.float32)
    yw = np.zeros((M,), dtype=np.int64)

    for i, end in enumerate(idx_end):
        start = end - (win - 1)
        Xw[i] = X[start : end + 1]
        yw[i] = int(y[end])
    return Xw, yw


class WindowsDataset(Dataset):
    def __init__(self, Xw: np.ndarray, y: np.ndarray):
        self.Xw = torch.from_numpy(Xw).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.Xw[idx], self.y[idx]


def train_one_stage(
    cfg: CFG,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    num_classes: int,
    stage_name: str,
):
    # windows
    Xw_tr, yw_tr = make_windows(X_tr, y_tr, cfg.win, cfg.stride)
    Xw_va, yw_va = make_windows(X_va, y_va, cfg.win, cfg.stride)

    train_loader = DataLoader(WindowsDataset(Xw_tr, yw_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(WindowsDataset(Xw_va, yw_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    feature_dim = Xw_tr.shape[-1]
    model = AnomalyDetectionModel(
        feature_dim=feature_dim,
        num_classes=num_classes,
        model_dim=cfg.model_dim,
        tcn_layers=cfg.tcn_layers,
        transformer_layers=cfg.transformer_layers,
        nheads=cfg.nheads,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        # val
        model.eval()
        va_loss = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += loss.item()
                pred = logits.argmax(dim=1)
                all_pred.extend(pred.detach().cpu().numpy().tolist())
                all_true.extend(yb.detach().cpu().numpy().tolist())

        va_f1 = f1_score(all_true, all_pred, average="macro")
        print(f"[{stage_name}] Epoch {ep:02d}/{cfg.epochs} | tr_loss={tr_loss/len(train_loader):.4f} | "
              f"va_loss={va_loss/len(val_loader):.4f} | va_macroF1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # restore best
    assert best_state is not None
    model.load_state_dict(best_state)
    return model


def eval_stage_binary(cfg: CFG, model, X_te: np.ndarray, y_te: np.ndarray) -> None:
    Xw_te, yw_te = make_windows(X_te, y_te, cfg.win, cfg.stride)
    loader = DataLoader(WindowsDataset(Xw_te, yw_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model.eval()
    probs, preds, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]  # P(Attack)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            probs.extend(p.detach().cpu().numpy().tolist())
            preds.extend(pred.tolist())
            trues.extend(yb.numpy().tolist())

    cm = confusion_matrix(trues, preds, labels=[0, 1])
    p, r, f1b, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
    auc = roc_auc_score(np.array(trues, dtype=np.int32), np.array(probs, dtype=np.float32))
    print("\n========== [Stage-1] Binary (Normal vs Attack) ==========")
    print("Confusion Matrix [[TN FP],[FN TP]]:\n", cm)
    print(f"P={p:.4f} R={r:.4f} F1={f1b:.4f} AUROC={auc:.4f}")
    print(classification_report(trues, preds, target_names=["Normal", "Attack"], zero_division=0))


def eval_stage_multiclass_attack_only(
    cfg: CFG, model, X_te_attack: np.ndarray, y_te_attack: np.ndarray, idx_to_name: Dict[int, str]
) -> None:
    Xw_te, yw_te = make_windows(X_te_attack, y_te_attack, cfg.win, cfg.stride)
    loader = DataLoader(WindowsDataset(Xw_te, yw_te), batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.device)
            logits = model(xb)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(yb.numpy().tolist())

    labels_all = list(range(len(idx_to_name)))
    macro_f1 = f1_score(trues, preds, average="macro", labels=labels_all)
    prec, rec, f1c, sup = precision_recall_fscore_support(
        trues, preds, labels=labels_all, average=None, zero_division=0
    )

    print("\n========== [Stage-2] Attack Multi-class (Attack-only eval) ==========")
    print(f"Macro-F1={macro_f1:.4f}  |  Attack-window-count={len(trues)}")
    print("[Per-Class]")
    for i in labels_all:
        print(f"  {i:02d} {idx_to_name[i]:<15s}  P={prec[i]:.3f} R={rec[i]:.3f} F1={f1c[i]:.3f}  N={int(sup[i])}")


# ============================================================
# 3) 主流程
# ============================================================
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    print("🚀 Loading TON-IoT CSV:", cfg.data_csv)
    df = pd.read_csv(cfg.data_csv)
    print("Rows:", len(df), "Cols:", len(df.columns))

    # 构建原始数值矩阵
    X_all, extra = build_feature_matrix(df, cfg)
    y_bin = np.array(extra["y_bin"], dtype=np.int64)
    type_all = np.array(extra["type"], dtype=str)

    # Split (stratify by binary label)
    X_tr, X_te, y_tr, y_te, type_tr, type_te = train_test_split(
        X_all, y_bin, type_all, test_size=cfg.test_ratio, random_state=cfg.seed, stratify=y_bin
    )
    X_tr, X_va, y_tr, y_va, type_tr, type_va = train_test_split(
        X_tr, y_tr, type_tr, test_size=cfg.val_ratio_in_train, random_state=cfg.seed, stratify=y_tr
    )

    # 标准化（只 fit 训练集）
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    # STL 残差：TON-IoT 含大量离散/类别特征，STL 只对“连续数值列”生效（避免把 hash/onehot 做平滑导致特征语义被抹掉）
    num_cols = list(extra.get("num_cols", []))
    # 经验上更“连续”的数值列（存在则用；不存在则退化为对全部 num_cols）
    cont_candidates = [
        "duration","src_bytes","dst_bytes","missed_bytes",
        "src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes",
        "http_trans_depth","http_request_body_len","http_response_body_len",
        "dns_rejected"
    ]
    stl_cols = [c for c in cont_candidates if c in num_cols]
    if not stl_cols:
        # 兜底：排除明显离散的端口/状态码等（仍比全量好）
        stl_cols = [c for c in num_cols if c not in ("src_port","dst_port","http_status_code")]
    stl_idx = [num_cols.index(c) for c in stl_cols if c in num_cols]

    X_tr_stl = apply_stl_subset(X_tr_s, stl_idx, cfg, fit_on="train")
    X_va_stl = apply_stl_subset(X_va_s, stl_idx, cfg, fit_on="val")
    X_te_stl = apply_stl_subset(X_te_s, stl_idx, cfg, fit_on="test")

    # PCA（只 fit 训练集）
    pca = PCA(n_components=min(cfg.pca_dim, X_tr_stl.shape[1]), random_state=cfg.seed)
    X_tr_p = pca.fit_transform(X_tr_stl).astype(np.float32)
    X_va_p = pca.transform(X_va_stl).astype(np.float32)
    X_te_p = pca.transform(X_te_stl).astype(np.float32)
    print(f"[PCA] {X_tr_stl.shape[1]} -> {X_tr_p.shape[1]} | explained_var={pca.explained_variance_ratio_.sum():.4f}")

    # ===========================
    # Stage-1: Binary
    # ===========================
    model_bin = train_one_stage(cfg, X_tr_p, y_tr, X_va_p, y_va, num_classes=2, stage_name="Stage-1(BIN)")
    eval_stage_binary(cfg, model_bin, X_te_p, y_te)

    # ===========================
    # Stage-2: Attack-only multi-class
    # ===========================
    # 只取 Attack 子集，排除 type=normal，保留所有 attack types（不合并）
    mask_tr_a = (y_tr == 1) & (type_tr != "normal")
    mask_va_a = (y_va == 1) & (type_va != "normal")
    mask_te_a = (y_te == 1) & (type_te != "normal")

    tr_types = type_tr[mask_tr_a].tolist()
    va_types = type_va[mask_va_a].tolist()
    te_types = type_te[mask_te_a].tolist()

    attack_types = stable_sorted_unique(tr_types + va_types + te_types)
    name_to_idx = {n: i for i, n in enumerate(attack_types)}
    idx_to_name = {i: n for n, i in name_to_idx.items()}

    print("\n[Stage-2] Attack types (kept, no merge):", attack_types)
    # 每类样本数统计（按“原始行”统计）
    cnt_tr = pd.Series(tr_types).value_counts().to_dict()
    cnt_te = pd.Series(te_types).value_counts().to_dict()
    print("[Stage-2] Attack row-counts (train/val split part):", cnt_tr)
    print("[Stage-2] Attack row-counts (test split part):", cnt_te)

    # 将 type -> 多分类 label
    y_tr_a = np.array([name_to_idx[t] for t in type_tr[mask_tr_a]], dtype=np.int64)
    y_va_a = np.array([name_to_idx[t] for t in type_va[mask_va_a]], dtype=np.int64)
    y_te_a = np.array([name_to_idx[t] for t in type_te[mask_te_a]], dtype=np.int64)

    X_tr_a = X_tr_p[mask_tr_a]
    X_va_a = X_va_p[mask_va_a]
    X_te_a = X_te_p[mask_te_a]

    K = len(attack_types)
    model_mc = train_one_stage(cfg, X_tr_a, y_tr_a, X_va_a, y_va_a, num_classes=K, stage_name="Stage-2(ATTACK-MC)")
    eval_stage_multiclass_attack_only(cfg, model_mc, X_te_a, y_te_a, idx_to_name)


if __name__ == "__main__":
    main()
