import os
import glob
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import torch.nn.functional as F
from sklearn.decomposition import PCA



from model_doubleGAN import AnomalyDetectionModel

# ====================== 基本配置 ======================
config = {
    "feature_dim": 16,        # 会在加载数据后自动更新
    "model_dim": 128,
    "tcn_layers": 2,
    "transformer_layers": 3,
    "nheads": 4,
    "dropout": 0.4,
    "max_len": 64,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 8,
    "gradient_clip": 1.0,
    "num_classes": 5,
    "window": 64,
    "stride": 64,
    "maj_threshold": 0.55,
    "aug_shift_max": 0,
    "aug_prob": 0.0,
    "pca_n_components": 0.95,  # 也可以改成 8/12 这种整数
    # 预处理模式
    "use_gan_pca": True,
    # GAN（窗口级 cGAN）
    "gan_noise_dim": 64,
    "gan_hidden": 256,
    "gan_epochs": 30,           # 先跑通用 30；效果不稳再加
    "gan_batch": 256,
    "gan_lr": 2e-4,
    "gan_max_train_windows": 20000,  # 防止窗口太多训练 GAN 太慢；按需调大/设为0不限制

}

# ====================== 固定随机种子 ======================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ====================== 读取 parquet 文件夹 ======================
def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    parquet_files = [
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if f.endswith(".parquet") and not f.endswith(".parquet.crc")
    ]
    print(f"📂 在 {folder_path} 中发现 {len(parquet_files)} 个有效 Parquet 文件")
    dfs = []
    for file in parquet_files:
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            dfs.append(df)
            print(f"✅ 已读取 {os.path.basename(file)}，包含 {len(df)} 条记录")
        except Exception as e:
            print(f"⚠️ 跳过 {file}: {e}")
    if not dfs:
        raise RuntimeError(f"{folder_path} 下没有合法 parquet 数据")
    merged = pd.concat(dfs, ignore_index=True)
    print(f"✅ 合并后总记录数: {len(merged)}")
    return merged

# ====================== 窗口数据集 ======================
class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, M: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.M = torch.from_numpy(M).bool()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.M[idx]

# ====================== A2 + B4 滑窗构造 ======================
def make_windows_A2(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
    feature_cols,
    window: int,
    stride: int,
    maj_thr: float,
    aug_shift_max: int,
    aug_prob: float,
    idx_to_name,
    rare_ids=None,
):
    if rare_ids is None:
        rare_ids = set()
    df = df.sort_values([id_col, time_col])
    groups = df.groupby(id_col)

    X_list, y_list, M_list = [], [], []
    num_classes = len(idx_to_name)

    for sid, g in groups:
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        lab = g[label_col].to_numpy(dtype=np.int64)
        L = len(g)
        if L == 0:
            continue
        s = 0
        while s < L:
            e = min(s + window, L)
            x_real = feat[s:e]
            y_real = lab[s:e]
            valid_len = e - s

            # padding + mask
            if valid_len < window:
                pad_len = window - valid_len
                pad = np.zeros((pad_len, feat.shape[1]), dtype=np.float32)
                x_win = np.concatenate([x_real, pad], axis=0)
                pad_mask = np.ones((window,), dtype=bool)
                pad_mask[:valid_len] = False
            else:
                x_win = x_real
                pad_mask = np.zeros((window,), dtype=bool)

            # A2 窗口标签：稀有类优先 + 多数投票
            if valid_len > 0:
                cnt = np.bincount(y_real.astype(np.int64), minlength=num_classes)
                present_rare = [(r, cnt[r]) for r in rare_ids if cnt[r] > 0]
                if present_rare:
                    # 窗口里只要有稀有类，就选出现次数最多的稀有类
                    y_win = max(present_rare, key=lambda z: z[1])[0]
                else:
                    ratio = cnt / (cnt.sum() + 1e-12)
                    if ratio.max() >= maj_thr:
                        y_win = int(np.argmax(cnt))
                    else:
                        # 不满足阈值也不丢窗口，退回出现最多的类
                        y_win = int(np.argmax(cnt))
            else:
                y_win = 0

            X_list.append(x_win)
            y_list.append(y_win)
            M_list.append(pad_mask)
            s += stride

    X_win = np.stack(X_list, axis=0) if X_list else np.zeros((0, window, len(feature_cols)), np.float32)
    y_win = np.asarray(y_list, dtype=np.int64) if y_list else np.zeros((0,), np.int64)
    M_win = np.stack(M_list, axis=0) if M_list else np.zeros((0, window), np.bool_)

    counts = dict((int(c), int(v)) for c, v in Counter(y_win).items())
    print("[A2] window counts before B4:", counts)

    # B4：对稀有类时间平移增强（只在训练集）
    if y_win.size > 0 and aug_shift_max > 0 and aug_prob > 0:
        median_cnt = int(np.median(list(counts.values()))) if counts else 0
        rare_classes = {c for c, v in counts.items() if c in (rare_ids or set()) and v < median_cnt}
        if rare_classes:
            X_aug, y_aug, M_aug = [], [], []
            rng = np.random.default_rng()
            for x, y, m in zip(X_list, y_list, M_list):
                if y in rare_classes and rng.random() < aug_prob:
                    valid_idx = np.where(m == False)[0]
                    if valid_idx.size == 0:
                        continue
                    s0, e0 = valid_idx[0], valid_idx[-1] + 1
                    k = int(rng.integers(1, aug_shift_max + 1))
                    seg = x[s0:e0].copy()
                    seg = np.roll(seg, shift=k, axis=0)
                    x2 = x.copy()
                    x2[s0:e0] = seg
                    X_aug.append(x2)
                    y_aug.append(y)
                    M_aug.append(m.copy())
            if X_aug:
                print(f"[B4] augmented {len(X_aug)} windows for rare classes {sorted(rare_classes)}")
                X_win = np.concatenate([X_win, np.stack(X_aug, axis=0)], axis=0)
                y_win = np.concatenate([y_win, np.asarray(y_aug, dtype=np.int64)], axis=0)
                M_win = np.concatenate([M_win, np.stack(M_aug, axis=0)], axis=0)

    print("[A2/B4] final windows:", X_win.shape, y_win.shape, M_win.shape)
    if y_win.size:
        counts2 = dict((int(c), int(v)) for c, v in Counter(y_win).items())
        print("[A2/B4] window counts after B4:", counts2)
    return X_win, y_win, M_win

class CGAN_G(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, y):
        y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        x = torch.cat([z, y_oh], dim=1)
        return self.net(x)

class CGAN_D(nn.Module):
    def __init__(self, num_classes: int, in_dim: int, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),  # logits
        )

    def forward(self, x, y):
        y_oh = F.one_hot(y, num_classes=self.num_classes).float()
        h = torch.cat([x, y_oh], dim=1)
        return self.net(h).squeeze(1)

def train_cgan_windows(X_flat: np.ndarray, y: np.ndarray, num_classes: int, cfg: dict):
    device = cfg["device"]
    noise_dim = cfg["gan_noise_dim"]
    hidden = cfg["gan_hidden"]
    epochs = cfg["gan_epochs"]
    batch = cfg["gan_batch"]
    lr = cfg["gan_lr"]

    X = torch.from_numpy(X_flat).float().to(device)
    Y = torch.from_numpy(y).long().to(device)

    G = CGAN_G(noise_dim, num_classes, X_flat.shape[1], hidden).to(device)
    D = CGAN_D(num_classes, X_flat.shape[1], hidden).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            xb = X[idx]
            yb = Y[idx]
            bs = xb.size(0)
            if bs < 8:
                continue

            # -------- Train D --------
            z = torch.randn(bs, noise_dim, device=device)
            x_fake = G(z, yb).detach()
            logit_real = D(xb, yb)
            logit_fake = D(x_fake, yb)

            lossD = bce(logit_real, torch.ones_like(logit_real)) + bce(logit_fake, torch.zeros_like(logit_fake))
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # -------- Train G --------
            z = torch.randn(bs, noise_dim, device=device)
            x_fake = G(z, yb)
            logit_fake = D(x_fake, yb)
            lossG = bce(logit_fake, torch.ones_like(logit_fake))

            optG.zero_grad()
            lossG.backward()
            optG.step()

        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[GAN] epoch {ep+1}/{epochs} | lossD={lossD.item():.4f} lossG={lossG.item():.4f}")

    return G

@torch.no_grad()
def cgan_sample(G: nn.Module, y: np.ndarray, cfg: dict, num_classes: int):
    device = cfg["device"]
    noise_dim = cfg["gan_noise_dim"]
    y_t = torch.from_numpy(y).long().to(device)
    z = torch.randn(y_t.size(0), noise_dim, device=device)
    x = G(z, y_t).cpu().numpy().astype(np.float32)
    return x

def gan_balance_train_windows(X_tr: np.ndarray, y_tr: np.ndarray, M_tr: np.ndarray, num_classes: int, cfg: dict):
    W = cfg["window"]
    D = X_tr.shape[2]
    full_idx = ~M_tr.any(axis=1)  # 只用完整窗口训练GAN/生成

    X_full = X_tr[full_idx].reshape(-1, W * D)
    y_full = y_tr[full_idx]

    # 可选：限制 GAN 训练样本数（防止太慢）
    cap = int(cfg.get("gan_max_train_windows", 0) or 0)
    if cap > 0 and X_full.shape[0] > cap:
        sel = np.random.choice(X_full.shape[0], cap, replace=False)
        X_gan = X_full[sel]
        y_gan = y_full[sel]
    else:
        X_gan = X_full
        y_gan = y_full

    cnt = Counter(y_full.tolist())
    target = max(cnt.values())
    print("[GAN] before:", dict(cnt), "| target:", target)

    G = train_cgan_windows(X_gan, y_gan, num_classes, cfg)

    synth_list = []
    synth_y_list = []
    for c in range(num_classes):
        need = target - cnt.get(c, 0)
        if need <= 0:
            continue
        y_need = np.full((need,), c, dtype=np.int64)
        x_need = cgan_sample(G, y_need, cfg, num_classes)
        synth_list.append(x_need)
        synth_y_list.append(y_need)

    if len(synth_list) == 0:
        print("[GAN] already balanced, skip generation.")
        return X_tr, y_tr, M_tr

    X_syn = np.concatenate(synth_list, axis=0).reshape(-1, W, D)
    y_syn = np.concatenate(synth_y_list, axis=0)
    M_syn = np.zeros((X_syn.shape[0], W), dtype=bool)

    # 把 padded 窗口原样拼回（不让 GAN 生成 padding）
    X_pad, y_pad, M_pad = X_tr[~full_idx], y_tr[~full_idx], M_tr[~full_idx]

    X_new = np.concatenate([X_tr[full_idx], X_syn, X_pad], axis=0)
    y_new = np.concatenate([y_tr[full_idx], y_syn, y_pad], axis=0)
    M_new = np.concatenate([M_tr[full_idx], M_syn, M_pad], axis=0)

    perm = np.random.permutation(len(y_new))
    X_new, y_new, M_new = X_new[perm], y_new[perm], M_new[perm]

    print("[GAN] after :", dict(Counter(y_new.tolist())))
    return X_new, y_new, M_new

def pca_transform_windows(X_tr: np.ndarray, M_tr: np.ndarray, X_va: np.ndarray, M_va: np.ndarray, cfg: dict):
    ncomp = cfg.get("pca_n_components", 0.95)
    D = X_tr.shape[2]

    # 只用“非padding”的 timestep 拟合 PCA
    Xtr_flat = X_tr.reshape(-1, D)
    Mtr_flat = M_tr.reshape(-1)
    X_fit = Xtr_flat[~Mtr_flat]

    pca = PCA(n_components=ncomp, random_state=42)
    pca.fit(X_fit)

    def _trans(X, M):
        N, W, D0 = X.shape
        flat = X.reshape(-1, D0)
        out = pca.transform(flat).astype(np.float32)
        out[M.reshape(-1)] = 0.0  # 强制 padding timestep 仍为 0
        return out.reshape(N, W, -1)

    X_tr_p = _trans(X_tr, M_tr)
    X_va_p = _trans(X_va, M_va)

    print(f"[PCA] {D} -> {X_tr_p.shape[2]} | explained_var={pca.explained_variance_ratio_.sum():.4f}")
    return X_tr_p, X_va_p, pca


# ====================== 主训练流程 ======================
def main():
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    start_time = datetime.now()
    run_tag = start_time.strftime("%Y%m%d-%H%M%S")
    log_path = save_dir / "experiments_log.txt"

    def append_log(text: str):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    append_log("")
    append_log(f"========== EXPERIMENT {run_tag} ==========")
    append_log(f"Started    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------ 1. 加载原始训练/测试数据 ------------
    print("🚀 开始加载 WSN-DS 数据集...")
    train_df = load_parquet_folder(r"C:\Users\yeqing\PycharmProjects\pythonProject\WSN-DS-main\train.parquet")
    test_df = load_parquet_folder(r"C:\Users\yeqing\PycharmProjects\pythonProject\WSN-DS-main\test.parquet")

    print("训练集攻击类型分布：\n", train_df["Attack_type"].value_counts())
    print("测试集攻击类型分布：\n", test_df["Attack_type"].value_counts())

    ID_COL = "id"
    TIME_COL = "Time"
    LABEL_COL = "Attack_type"
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TIME_COL, LABEL_COL]]

    print("🔧 数据预处理（GAN+PCA：先标准化，后窗口GAN，最后PCA）...")

    # 1) 缺失值填充
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    # 2) 标准化（只在训练集 fit）
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))



    # ------------ 3. 标签编码成 0~4 ------------
    le = LabelEncoder()
    le.fit(pd.concat([train_df[LABEL_COL], test_df[LABEL_COL]], ignore_index=True))
    train_df[LABEL_COL] = le.transform(train_df[LABEL_COL])
    test_df[LABEL_COL] = le.transform(test_df[LABEL_COL])
    idx_to_name = [str(x) for x in le.classes_]
    num_classes = len(idx_to_name)
    config["num_classes"] = num_classes
    print("编码后的类别：", idx_to_name)

    # ——重点：把所有攻击类(非0)都当成 rare_ids，A2/B4 里优先照顾——
    rare_ids = {c for c in range(num_classes) if c != 0}

    # === 额外把 1 和 3 也当成“需要照顾”的类 ===
    for c in [1, 3]:
        rare_ids.add(c)

    print("[Classes]", num_classes, "=>", idx_to_name, "| rare_ids:", rare_ids)

    # ------------ 4. 按 A2+B4 构造训练/验证窗口 ------------
    W = config["window"]
    ST = config["stride"]
    TH = config["maj_threshold"]

    X_tr, y_tr, M_tr = make_windows_A2(
        train_df, ID_COL, TIME_COL, LABEL_COL, feature_cols,
        W, ST, TH,
        config["aug_shift_max"], config["aug_prob"],
        idx_to_name, rare_ids=rare_ids,
    )
    X_va, y_va, M_va = make_windows_A2(
        test_df, ID_COL, TIME_COL, LABEL_COL, feature_cols,
        W, ST, TH,
        0, 0.0,
        idx_to_name, rare_ids=rare_ids,
    )

    # ------------ 4.1 GAN：对训练窗口做生成补齐（仅训练集）------------
    if config.get("use_gan_pca", False):
        X_tr, y_tr, M_tr = gan_balance_train_windows(X_tr, y_tr, M_tr, num_classes, config)

    # ------------ 4.2 PCA：在（GAN增强后的训练窗口）上fit，再transform训练/验证 ------------
    if config.get("use_gan_pca", False):
        X_tr, X_va, _ = pca_transform_windows(X_tr, M_tr, X_va, M_va, config)


    print("[Final window counts]", dict(Counter(y_tr.tolist())))
    config["feature_dim"] = X_tr.shape[2]
    print(f"特征维度更新为: {config['feature_dim']}")

    train_dataset = WindowsDataset(X_tr, y_tr, M_tr)
    val_dataset = WindowsDataset(X_va, y_va, M_va)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    gc.collect()

    # ------------ 5. 构建模型 ------------
    model = AnomalyDetectionModel(
        feature_dim=config["feature_dim"],
        model_dim=config["model_dim"],
        num_classes=config["num_classes"],
        tcn_layers=config["tcn_layers"],
        transformer_layers=config["transformer_layers"],
        nheads=config["nheads"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    ).to(config["device"])

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------ 6. 优化器 + 调度器 + 损失函数 ------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["learning_rate"] * 0.1,
    )

    # ——多分类 CrossEntropy + 类别权重（就是你 best 那版的核心）——
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_tr,
    ).astype(np.float32)

    # === 手动给 1 和 3 再加一点权重（例如 +20%） ===
    #for c in [1, 3]:
    #    class_weights[c] *= 1.2  # 可以先试 1.2，不要太大，后面再看效果微调

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=config["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print("[class_weights]", {i: float(w) for i, w in enumerate(class_weights)})

    # ------------ 7. 训练循环 ------------
    best_f1 = 0.0
    best_epoch = -1
    no_improve = 0
    all_train_losses, all_val_losses = [], []
    all_train_f1s, all_val_f1s = [], []

    labels_all = np.arange(num_classes)

    for epoch in range(config["epochs"]):
        # === Train ===
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        from tqdm.auto import tqdm
        for inputs, labels, pad_mask in tqdm(
            train_loader,
            desc=f"[Train] {epoch+1}/{config['epochs']}",
            leave=False,
            dynamic_ncols=True,
        ):
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])
            pad_mask = pad_mask.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs, pad_mask=pad_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["gradient_clip"])
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())

        train_avg_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_targets, train_preds, average="macro", labels=labels_all, zero_division=0)
        print(f"Epoch {epoch+1} - 训练 - 损失: {train_avg_loss:.4f}, F1: {train_f1:.4f}")
        print("训练集混淆矩阵:\n", confusion_matrix(train_targets, train_preds))
        all_train_losses.append(train_avg_loss)
        all_train_f1s.append(train_f1)

        # === Val ===
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels, pad_mask in val_loader:
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])
                pad_mask = pad_mask.to(config["device"])

                outputs = model(inputs, pad_mask=pad_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_targets.extend(labels.detach().cpu().numpy().tolist())

        val_avg_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average="macro", labels=labels_all, zero_division=0)
        print(f"Epoch {epoch+1}/{config['epochs']} - 验证 - 损失: {val_avg_loss:.4f}, F1: {val_f1:.4f}")
        print("验证集混淆矩阵:\n", confusion_matrix(val_targets, val_preds))

        prec, rec, f1c, sup = precision_recall_fscore_support(
            val_targets, val_preds, labels=labels_all, average=None, zero_division=0
        )
        print("\n[Per-Class on VAL]")
        for i in range(num_classes):
            name = idx_to_name[i] if i < len(idx_to_name) else f"class_{i}"
            print(f"{name:<5}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1c[i]:.3f}  N={int(sup[i])}")

        scheduler.step()
        all_val_losses.append(val_avg_loss)
        all_val_f1s.append(val_f1)

        # 早停 + 保存最优
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": val_f1,
                    "config": config,
                },
                save_dir / "best_model.pth",
            )
            print(f"💾 保存最佳模型，F1: {best_f1:.4f}")
        else:
            no_improve += 1
            print(f"⏸ 模型性能未提升 ({no_improve}/{config['patience']})")
            if no_improve >= config["patience"]:
                print(f"🛑 提前停止：{config['patience']} 个 epoch 未改善")
                break

    # 保存训练曲线
    np.savez(
        save_dir / "training_history.npz",
        train_losses=np.array(all_train_losses),
        val_losses=np.array(all_val_losses),
        train_f1s=np.array(all_train_f1s),
        val_f1s=np.array(all_val_f1s),
    )
    print(f"🎯 训练完成！最佳验证 F1: {best_f1:.4f}, epoch={best_epoch}")

if __name__ == "__main__":
    main()
