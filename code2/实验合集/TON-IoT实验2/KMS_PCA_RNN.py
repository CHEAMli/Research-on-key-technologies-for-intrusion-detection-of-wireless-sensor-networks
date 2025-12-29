import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from typing import Tuple, Dict, List


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAIN_FILE = os.path.join(BASE_DIR, "train_no_injection_with_origidx.csv")
TEST_FILE = os.path.join(BASE_DIR, "test_full_with_injection_with_origidx.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
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
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, F]
        out, _ = self.rnn(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits


def prepare_data(n_clusters: int = 10, pca_components: int = 10) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], LabelEncoder]:
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Numeric features only
    numeric_features = ['src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes', 
                       'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
                       'dns_qclass', 'dns_qtype', 'dns_rcode', 
                       'http_request_body_len', 'http_response_body_len', 'http_status_code']
    
    # Encode type labels
    le = LabelEncoder()
    all_types = pd.concat([train_df['type'], test_df['type']])
    le.fit(all_types)
    
    print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Prepare features and labels
    X_train = train_df[numeric_features].fillna(0).values.astype(np.float32)
    y_train = le.transform(train_df['type']).astype(np.int64)
    type_train = train_df['type'].values
    
    X_test = test_df[numeric_features].fillna(0).values.astype(np.float32)
    y_test = le.transform(test_df['type']).astype(np.int64)
    type_test = test_df['type'].values
    
    # Step 1: StandardScaler
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Step 2: KMeans clustering
    print(f"Applying KMeans clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)
    
    # Add cluster as feature
    X_train_with_cluster = np.column_stack([X_train, train_clusters])
    X_test_with_cluster = np.column_stack([X_test, test_clusters])
    
    # Step 3: PCA
    print(f"Applying PCA (n_components={pca_components})...")
    pca = PCA(n_components=pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_with_cluster)
    X_test_pca = pca.transform(X_test_with_cluster)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return (X_train_pca, y_train, type_train), (X_test_pca, y_test, type_test), le


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    batch_size: int = 256,
    epochs: int = 1,
    lr: float = 1e-3,
    rnn_type: str = "lstm",
) -> RNNClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(input_dim, hidden_dim, num_layers, num_classes, rnn_type=rnn_type).to(device)
    ds = TabularDataset(X_train, y_train)
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
    ds = TabularDataset(X, np.zeros((X.shape[0],), dtype=np.int64))
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
    epochs = int(os.environ.get("EPOCHS", 1))
    hidden_dim = int(os.environ.get("HIDDEN", 64))
    rnn_type = os.environ.get("RNN", "lstm")
    n_clusters = int(os.environ.get("N_CLUSTERS", 10))
    pca_components = int(os.environ.get("PCA_COMPONENTS", 10))
    unknown_threshold = float(os.environ.get("UNKNOWN_THRESH", 0.6))
    
    # Prepare data
    (X_train, y_train, type_train), (X_test, y_test, type_test), le = prepare_data(n_clusters=n_clusters, pca_components=pca_components)
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    
    print(f"\nFeatures: {input_dim}, Classes: {num_classes}")
    print(f"Training model (RNN with KMeans+PCA, rnn_type={rnn_type})...")
    
    model = train_model(X_train, y_train, input_dim, num_classes, hidden_dim=hidden_dim, epochs=epochs, rnn_type=rnn_type)
    
    print("Predicting on test set...")
    probs = predict_proba(model, X_test)
    y_pred = probs.argmax(axis=1)
    y_maxp = probs.max(axis=1)
    
    # Find injection label index
    injection_idx = list(le.classes_).index('injection')
    normal_idx = list(le.classes_).index('normal')
    
    # 1. Attack vs Normal 二分类评估（排除injection）
    mask_known = type_test != 'injection'
    y_true_known = y_test[mask_known]
    y_pred_known = y_pred[mask_known]
    probs_known = probs[mask_known]
    
    y_true_binary = (y_true_known != normal_idx).astype(int)
    y_pred_binary = (y_pred_known != normal_idx).astype(int)
    prob_attack = 1 - probs_known[:, normal_idx]
    
    binary_p, binary_r, binary_f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
    binary_auroc = roc_auc_score(y_true_binary, prob_attack)
    
    print("\n===== Attack vs Normal 二分类评估（排除injection）=====")
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
    pd.DataFrame([binary_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_kms_pca_binary_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # 2. 多分类评估（排除injection）
    type_names_known = [le.classes_[i] for i in sorted(set(y_true_known))]
    
    report = classification_report(y_true_known, y_pred_known, labels=sorted(set(y_true_known)), 
                                   target_names=type_names_known, digits=4, output_dict=False, zero_division=0)
    print("\n===== 多分类评估（排除injection）=====")
    print(report)
    with open(os.path.join(RESULTS_DIR, "rnn_kms_pca_known_class_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    report_dict = classification_report(y_true_known, y_pred_known, labels=sorted(set(y_true_known)), output_dict=True, zero_division=0)
    macro_f1 = report_dict['macro avg']['f1-score']
    
    # Coverage: 只在攻击类上计算
    mask_attacks = y_true_known != normal_idx
    y_true_attacks = y_true_known[mask_attacks]
    probs_attacks = probs_known[mask_attacks]
    
    attack_classes = sorted([i for i in set(y_true_known) if i != normal_idx])
    n_classes = len(attack_classes)
    label_to_idx = {label: idx for idx, label in enumerate(attack_classes)}
    y_true_remapped = np.array([label_to_idx[y] for y in y_true_attacks])
    probs_attacks_only = probs_attacks[:, attack_classes]
    
    rankings = []
    for i in range(len(y_true_remapped)):
        true_label_idx = y_true_remapped[i]
        sorted_indices = np.argsort(probs_attacks_only[i])[::-1]
        rank = np.where(sorted_indices == true_label_idx)[0][0] + 1
        rankings.append(rank)
    average_rank = np.mean(rankings)
    coverage = 1.0 / average_rank
    
    # 严格评估：考虑injection的干扰
    mask_injection = type_test == 'injection'
    injection_pred = y_pred[mask_injection]
    injection_maxp = y_maxp[mask_injection]
    
    strict_f1_scores = []
    known_labels = sorted(set(y_true_known))
    for true_label in known_labels:
        mask_true = y_true_known == true_label
        mask_pred = y_pred_known == true_label
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        inj_misclassified = int(((injection_pred == true_label) & (injection_maxp >= unknown_threshold)).sum())
        fp += inj_misclassified
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        strict_f1_scores.append(f1)
    strict_macro_f1 = np.mean(strict_f1_scores)
    
    strict_rankings = rankings.copy()
    for i in range(len(injection_pred)):
        if injection_maxp[i] >= unknown_threshold and injection_pred[i] in attack_classes:
            strict_rankings.append(n_classes)
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
    pd.DataFrame([multiclass_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_kms_pca_multiclass_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # 3. Unknown（Injection）拒识性能评估
    # 只在攻击样本上评估（排除normal）
    mask_attacks_all = type_test != 'normal'
    y_test_attacks = y_test[mask_attacks_all]
    y_maxp_attacks = y_maxp[mask_attacks_all]
    type_attacks = type_test[mask_attacks_all]
    
    y_true_unknown_attacks = (type_attacks == 'injection').astype(int)
    y_pred_unknown_attacks = (y_maxp_attacks < unknown_threshold).astype(int)
    prob_unknown_attacks = 1 - y_maxp_attacks
    
    unknown_p, unknown_r, unknown_f1, _ = precision_recall_fscore_support(y_true_unknown_attacks, y_pred_unknown_attacks, average='binary', zero_division=0)
    unknown_auroc = roc_auc_score(y_true_unknown_attacks, prob_unknown_attacks)
    
    print("\n===== Unknown（Injection）拒识性能（仅攻击类）=====")
    print(f"Unknown-Precision: {unknown_p:.4f}")
    print(f"Unknown-Recall:    {unknown_r:.4f}")
    print(f"Unknown-F1:        {unknown_f1:.4f}")
    print(f"Unknown-AUROC:     {unknown_auroc:.4f}")
    print(f"\n[Diagnostic] Injection maxp stats: min={injection_maxp.min():.4f}, mean={injection_maxp.mean():.4f}, max={injection_maxp.max():.4f}")
    print(f"[Diagnostic] Attacks only: {len(y_test_attacks)} samples, {int(y_true_unknown_attacks.sum())} injection")
    
    unknown_metrics = {
        "Unknown-Precision": unknown_p,
        "Unknown-Recall": unknown_r,
        "Unknown-F1": unknown_f1,
        "Unknown-AUROC": unknown_auroc
    }
    pd.DataFrame([unknown_metrics]).to_csv(os.path.join(RESULTS_DIR, "rnn_kms_pca_unknown_metrics.csv"), index=False, encoding="utf-8-sig")
    
    # 4. Injection样本去向分析
    total_injection = int(mask_injection.sum())
    
    # Count by category
    pred_unknown = int((injection_maxp < unknown_threshold).sum())
    
    # Normal class
    normal_idx = list(le.classes_).index('normal')
    pred_normal = int(((injection_pred == normal_idx) & (injection_maxp >= unknown_threshold)).sum())
    
    # Known attacks (exclude normal and injection)
    pred_known = 0
    for i, cls_name in enumerate(le.classes_):
        if cls_name not in ['injection', 'normal']:
            pred_known += int(((injection_pred == i) & (injection_maxp >= unknown_threshold)).sum())
    
    summary: Dict[str, int] = {
        "Injection_total": total_injection,
        "Injection_to_Normal": pred_normal,
        "Injection_to_Unknown": pred_unknown,
        "Injection_to_Known": pred_known,
    }
    
    print("\n[Unknown Eval][Injection as Unknown]")
    print(f"Injection windows total: {total_injection}")
    pct_normal = (pred_normal / total_injection * 100.0) if total_injection else 0.0
    pct_unknown = (pred_unknown / total_injection * 100.0) if total_injection else 0.0
    pct_known = (pred_known / total_injection * 100.0) if total_injection else 0.0
    print(f"  Injection→Normal : {pred_normal} ({pct_normal:.2f}%)")
    print(f"  Injection→Unknown: {pred_unknown} ({pct_unknown:.2f}%)")
    print(f"  Injection→Known  : {pred_known} ({pct_known:.2f}%)")
    
    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, "rnn_kms_pca_injection_routing_summary.csv"), index=False, encoding="utf-8-sig")
    
    print("\nArtifacts saved to:")
    print("-", os.path.join(RESULTS_DIR, "rnn_kms_pca_binary_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_kms_pca_multiclass_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_kms_pca_unknown_metrics.csv"))
    print("-", os.path.join(RESULTS_DIR, "rnn_kms_pca_known_class_report.txt"))
    print("-", os.path.join(RESULTS_DIR, "rnn_kms_pca_injection_routing_summary.csv"))


if __name__ == "__main__":
    main()
