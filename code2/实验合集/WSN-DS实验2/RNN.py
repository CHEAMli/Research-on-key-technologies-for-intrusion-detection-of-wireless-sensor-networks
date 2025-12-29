import os
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, coverage_error, precision_recall_fscore_support


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_DIR = os.path.join(BASE_DIR, "newtrain.parquet")
TEST_DIR = os.path.join(BASE_DIR, "test.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_parquet_dir(dir_path: str) -> pd.DataFrame:
    files: List[str] = [
        f for f in glob.glob(os.path.join(dir_path, "*.parquet")) if os.path.isfile(f)
    ]
    if not files:
        raise FileNotFoundError(f"No .parquet files found in: {dir_path}")
    tables = [pq.read_table(f) for f in sorted(files)]
    table = pa.concat_tables(tables)
    return table.to_pandas()


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
    label_col: str = "Attack_type",
    group_col: str = "id",
    time_col: str = "Time",
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, g in df.groupby(group_col):
        g = g.sort_values(time_col)
        feat = g[feature_cols].values.astype(np.float32)
        labels = g[label_col].values.astype(np.int64)
        if len(g) < seq_len:
            continue
        # sliding window with stride=1
        for i in range(len(g) - seq_len + 1):
            X_list.append(feat[i : i + seq_len])
            y_list.append(int(labels[i + seq_len - 1]))

    if not X_list:
        raise RuntimeError("No sequences constructed; check seq_len or grouping.")

    X = np.stack(X_list, axis=0)  # [N, T, F]
    y = np.array(y_list, dtype=np.int64)  # [N]
    return X, y


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, rnn_type: str = "lstm"):
        super().__init__()
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else 0.1)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else 0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):  # x: [B, T, F]
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits


def prepare_data(seq_len: int = 8) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    train_df = load_parquet_dir(TRAIN_DIR)
    test_df = load_parquet_dir(TEST_DIR)

    # Columns
    label_col = "Attack_type"
    id_col = "id"
    time_col = "Time"
    
    # 统计原始数据分布
    print("\n原始测试集标签分布（序列构建前）:")
    print(test_df[label_col].value_counts().sort_index())

    # Decide feature columns: drop label and non-feature identifiers (id). Keep the rest numeric
    drop_cols = {label_col, id_col}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Fit scaler on train, transform both
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].astype(np.float32))
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols].astype(np.float32))

    # Build sequences
    X_train, y_train = build_sequences(train_df_scaled, seq_len, feature_cols, label_col=label_col, group_col=id_col, time_col=time_col)
    X_test, y_test = build_sequences(test_df_scaled, seq_len, feature_cols, label_col=label_col, group_col=id_col, time_col=time_col)
    
    print("\n序列构建后的测试集标签分布:")
    from collections import Counter
    print(pd.Series(Counter(y_test)).sort_index())

    return (X_train, y_train), (X_test, y_test), feature_cols


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    num_classes: int = 5,
    hidden_dim: int = 64,
    num_layers: int = 1,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3,
    rnn_type: str = "lstm",
) -> RNNClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(input_dim, hidden_dim, num_layers, num_classes, rnn_type=rnn_type).to(device)
    ds = SeqDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
        print(f"Epoch {ep}/{epochs} - loss: {total_loss/total:.4f}")

    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    ds = SeqDataset(X, np.zeros((X.shape[0],), dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_list: List[np.ndarray] = []
    softmax = nn.Softmax(dim=1)
    for xb, _ in dl:
        xb = xb.to(device)
        logits = model(xb)
        probs = softmax(logits).cpu().numpy()
        probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)


def main():
    # Hyperparameters
    seq_len = int(os.environ.get("SEQ_LEN", 8))
    epochs = int(os.environ.get("EPOCHS", 1))
    hidden_dim = int(os.environ.get("HIDDEN", 64))
    rnn_type = os.environ.get("RNN", "lstm")
    unknown_threshold = float(os.environ.get("UNKNOWN_THRESH", 0.6))

    print("Loading and preparing data...")
    (X_train, y_train), (X_test, y_test), feature_cols = prepare_data(seq_len=seq_len)
    input_dim = X_train.shape[2]
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}, Features: {input_dim}")

    # Sanity check: newtrain 不含灰洞(2)
    unique_train = sorted(set(y_train.tolist()))
    print("Train label set:", unique_train)

    print("Training model (RNN)...")
    model = train_model(X_train, y_train, input_dim=input_dim, num_classes=5, hidden_dim=hidden_dim, epochs=epochs, rnn_type=rnn_type)

    print("Predicting on test set...")
    probs = predict_proba(model, X_test)
    y_pred = probs.argmax(axis=1)
    y_maxp = probs.max(axis=1)

    # 1. Attack vs Normal 二分类评估（排除灰洞）
    mask_4cls = y_test != 2
    y_true_4cls = y_test[mask_4cls]
    y_pred_4cls = y_pred[mask_4cls]
    probs_4cls = probs[mask_4cls]
    
    # 将所有攻击类型合并为1，Normal为0
    y_true_binary = (y_true_4cls != 0).astype(int)
    y_pred_binary = (y_pred_4cls != 0).astype(int)
    # 攻击概率 = 1 - Normal概率
    prob_attack = 1 - probs_4cls[:, 0]
    
    binary_p, binary_r, binary_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    binary_auroc = roc_auc_score(y_true_binary, prob_attack)
    
    print("\n===== Attack vs Normal 二分类评估（排除灰洞）=====")
    print(f"Precision: {binary_p:.4f}")
    print(f"Recall:    {binary_r:.4f}")
    print(f"F1-Score:  {binary_f1:.4f}")
    print(f"AUROC:     {binary_auroc:.4f}")
    
    binary_metrics = {
        "Precision": binary_p,
        "Recall": binary_r,
        "F1": binary_f1,
        "AUROC": binary_auroc
    }
    pd.DataFrame([binary_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_binary_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # 2. 四分类评估（排除灰洞标签=2）
    report = classification_report(y_true_4cls, y_pred_4cls, labels=[0, 1, 3, 4], target_names=["Normal(0)", "Blackhole(1)", "Flooding(3)", "TDMA/Scheduling(4)"], digits=4, output_dict=False)
    print("\n===== 四分类评估（排除灰洞）=====")
    print(report)
    with open(os.path.join(RESULTS_DIR, "rnn_four_class_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    # 提取Macro-F1和计算Coverage
    report_dict = classification_report(y_true_4cls, y_pred_4cls, labels=[0, 1, 3, 4], output_dict=True, zero_division=0)
    macro_f1 = report_dict['macro avg']['f1-score']
    
    # Coverage: 只在攻击类上计算（排除Normal），避免被Normal主导
    mask_attacks_4cls = y_true_4cls != 0
    y_true_attacks = y_true_4cls[mask_attacks_4cls]
    probs_attacks = probs_4cls[mask_attacks_4cls]
    
    label_map = {0: 0, 1: 1, 3: 2, 4: 3}
    y_true_remapped = np.array([label_map[y] for y in y_true_attacks])
    # 提取攻击类的概率（排除Normal列）
    probs_attacks_only = np.column_stack([probs_attacks[:, 1], probs_attacks[:, 3], probs_attacks[:, 4]])  # Blackhole, Flooding, TDMA
    n_classes = 3  # 3个攻击类
    
    rankings = []
    for i in range(len(y_true_remapped)):
        true_label_mapped = y_true_remapped[i] - 1  # 1->0, 2->1, 3->2
        sorted_indices = np.argsort(probs_attacks_only[i])[::-1]
        rank = np.where(sorted_indices == true_label_mapped)[0][0] + 1
        rankings.append(rank)
    average_rank = np.mean(rankings)
    # MRR (Mean Reciprocal Rank) 归一化: 1/平均排名
    coverage = 1.0 / average_rank
    
    # 严格评估：考虑grayhole的干扰
    mask_gray = y_test == 2
    gray_pred = y_pred[mask_gray]
    gray_maxp = y_maxp[mask_gray]
    
    # 严格Macro-F1: 将高置信度误分的grayhole计入FP
    strict_f1_scores = []
    for true_label in [0, 1, 3, 4]:
        mask_true = y_true_4cls == true_label
        mask_pred = y_pred_4cls == true_label
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        
        # 添加grayhole被高置信度误分为该类的数量
        gray_misclassified = int(((gray_pred == true_label) & (gray_maxp >= unknown_threshold)).sum())
        fp += gray_misclassified
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        strict_f1_scores.append(f1)
    
    strict_macro_f1 = np.mean(strict_f1_scores)
    
    # 严格Coverage: 将grayhole被误分的情况也纳入
    strict_rankings = rankings.copy()
    for i in range(len(gray_pred)):
        if gray_maxp[i] >= unknown_threshold and gray_pred[i] in [1, 3, 4]:
            strict_rankings.append(n_classes)  # 最差排名
    strict_average_rank = np.mean(strict_rankings)
    strict_coverage = 1.0 / strict_average_rank
    
    print(f"\nMacro-F1:  {strict_macro_f1:.4f}")
    print(f"Coverage:  {strict_coverage:.4f}")
    print(f"[Diagnostic] Samples: {len(y_true_attacks)} attacks, rank={average_rank:.2f}, strict_rank={strict_average_rank:.2f}")
    
    multiclass_metrics = {
        "Macro-F1": macro_f1,
        "Coverage": coverage,
        "Strict_Macro-F1": strict_macro_f1,
        "Strict_Coverage": strict_coverage
    }
    pd.DataFrame([multiclass_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_multiclass_metrics.csv"), index=False, encoding="utf-8-sig")

    # 3. Unknown（Grayhole）拒识性能评估
    # 只在攻击样本上评估（排除Normal），避免被大量Normal主导
    mask_attacks = y_test != 0  # 所有政击类别（包括grayhole）
    y_test_attacks = y_test[mask_attacks]
    y_maxp_attacks = y_maxp[mask_attacks]
    
    # 在攻击样本上计算Unknown指标
    y_true_unknown_attacks = (y_test_attacks == 2).astype(int)  # grayhole=1, 其他攻击=0
    y_pred_unknown_attacks = (y_maxp_attacks < unknown_threshold).astype(int)
    prob_unknown_attacks = 1 - y_maxp_attacks
    
    unknown_p, unknown_r, unknown_f1, _ = precision_recall_fscore_support(y_true_unknown_attacks, y_pred_unknown_attacks, average='binary', zero_division=0)
    unknown_auroc = roc_auc_score(y_true_unknown_attacks, prob_unknown_attacks)
    
    print("\n===== Unknown（Grayhole）拒识性能（仅攻击类）=====")
    print(f"Unknown-Precision: {unknown_p:.4f}")
    print(f"Unknown-Recall:    {unknown_r:.4f}")
    print(f"Unknown-F1:        {unknown_f1:.4f}")
    print(f"Unknown-AUROC:     {unknown_auroc:.4f}")
    print(f"\n[Diagnostic] Grayhole maxp stats: min={gray_maxp.min():.4f}, mean={gray_maxp.mean():.4f}, max={gray_maxp.max():.4f}")
    print(f"[Diagnostic] Attacks only: {len(y_test_attacks)} samples, {int(y_true_unknown_attacks.sum())} grayhole")
    
    unknown_metrics = {
        "Unknown-Precision": unknown_p,
        "Unknown-Recall": unknown_r,
        "Unknown-F1": unknown_f1,
        "Unknown-AUROC": unknown_auroc
    }
    pd.DataFrame([unknown_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_unknown_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # 4. 灰洞样本去向分析
    pred_unknown = int((gray_maxp < unknown_threshold).sum())
    pred_grayhole = int(((gray_pred == 2) & (gray_maxp >= unknown_threshold)).sum())
    pred_normal = int(((gray_pred == 0) & (gray_maxp >= unknown_threshold)).sum())
    pred_attack1 = int(((gray_pred == 1) & (gray_maxp >= unknown_threshold)).sum())
    pred_attack3 = int(((gray_pred == 3) & (gray_maxp >= unknown_threshold)).sum())
    pred_attack4 = int(((gray_pred == 4) & (gray_maxp >= unknown_threshold)).sum())

    total_gray = int(mask_gray.sum())
    summary: Dict[str, int] = {
        "Grayhole": total_gray,
        "pred_unknown": pred_unknown,
        "pred_Grayhole(2)": pred_grayhole,
        "pred_Normal(0)": pred_normal,
        "pred_Blackhole(1)": pred_attack1,
        "pred_Flooding(3)": pred_attack3,
        "pred_TDMA/Scheduling(4)": pred_attack4,
    }

    print("\n===== 灰洞样本去向（基于阈值 unknown）=====")
    for k, v in summary.items():
        pct = (v / total_gray * 100.0) if total_gray else 0.0
        print(f"{k}: {v} ({pct:.2f}%)")

    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, "rnn_grayhole_routing_summary.csv"), index=False, encoding="utf-8-sig")

    print("\nArtifacts saved to:")
    print("-", os.path.join(RESULTS_DIR, "rnn_binary_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_four_class_report.txt"))
    print("-", os.path.join(RESULTS_DIR, "rnn_multiclass_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_unknown_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_grayhole_routing_summary.csv"))


if __name__ == "__main__":
    main()
