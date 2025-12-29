import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, Dict, List

TRAIN_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\train_no_injection_with_origidx.csv"
TEST_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\test_full_with_injection_with_origidx.csv"
RESULTS_DIR = "D:\\workspace\\TTS-XGB\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_data(n_clusters: int = 10, pca_components: int = 10) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], LabelEncoder]:
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


def train_model(X_train: np.ndarray, y_train: np.ndarray, num_classes: int, n_estimators: int = 100,
                max_depth: int = None) -> RandomForestClassifier:
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    # Hyperparameters
    n_clusters = int(os.environ.get("N_CLUSTERS", 10))
    pca_components = int(os.environ.get("PCA_COMPONENTS", 10))
    unknown_threshold = 0.6  # Threshold for classifying as "Unknown"

    # Prepare data
    (X_train, y_train, type_train), (X_test, y_test, type_test), le = prepare_data(
        n_clusters=n_clusters, pca_components=pca_components
    )

    print(f"\nFeatures: {X_train.shape[1]}, Classes: {len(le.classes_)}")

    # Train the Random Forest model
    model = train_model(X_train, y_train, num_classes=len(le.classes_))

    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # Get probability scores for all classes

    # Get class indices
    normal_idx = list(le.classes_).index('normal')
    injection_idx = list(le.classes_).index('injection')
    known_attack_classes = [cls for cls in le.classes_ if cls not in ['normal', 'injection']]
    known_attack_indices = [le.transform([cls])[0] for cls in known_attack_classes]

    # ==============================================
    # A) 二分类评估 (normal vs 其他)
    # ==============================================
    print("\n===== A) 二分类评估 (normal vs 其他) =====")
    # 真实标签: normal=0, 其他=1
    y_binary_true = np.where(type_test == 'normal', 0, 1)
    # 预测标签: normal=0, 其他=1
    y_binary_pred = np.where(y_pred == normal_idx, 0, 1)
    # 正例概率 (其他类的概率)
    y_binary_proba = 1 - y_proba[:, normal_idx]

    # 计算指标
    binary_precision = precision_score(y_binary_true, y_binary_pred)
    binary_recall = recall_score(y_binary_true, y_binary_pred)
    binary_f1 = f1_score(y_binary_true, y_binary_pred)
    binary_auc = roc_auc_score(y_binary_true, y_binary_proba)
    cm = confusion_matrix(y_binary_true, y_binary_pred)

    print(f"精确率 (Precision): {binary_precision:.4f}")
    print(f"召回率 (Recall): {binary_recall:.4f}")
    print(f"F1分数: {binary_f1:.4f}")
    print(f"AUROC: {binary_auc:.4f}")
    print("混淆矩阵:")
    print(cm)
    print("混淆矩阵解释:")
    print(f"TN (正常预测为正常): {cm[0, 0]}")
    print(f"FP (正常预测为异常): {cm[0, 1]}")
    print(f"FN (异常预测为正常): {cm[1, 0]}")
    print(f"TP (异常预测为异常): {cm[1, 1]}")

    # ==============================================
    # B) 已知攻击多分类评估 (除injection外的攻击)
    # ==============================================
    print("\n===== B) 已知攻击多分类评估 =====")
    # 筛选已知攻击样本 (排除normal和injection)
    mask_known_attack = np.isin(type_test, known_attack_classes)

    if mask_known_attack.sum() > 0:
        y_known_true = y_test[mask_known_attack]
        y_known_pred = y_pred[mask_known_attack]

        # 计算Macro-F1
        macro_f1_known = f1_score(y_known_true, y_known_pred, average='macro', zero_division=0)

        # 修正：coverage_known 定义为 已知攻击中预测概率≥0.9的样本占比（符合需求）
        known_attack_proba = np.max(y_proba[mask_known_attack], axis=1)
        coverage_known = np.mean(known_attack_proba >= 0.90)

        print(f"Macro-F1 (已知攻击): {macro_f1_known:.4f}")
        print(f"覆盖率 (coverage_known, ≥0.90): {coverage_known:.4f}")
    else:
        print("无已知攻击样本用于评估")
        macro_f1_known = 0.0
        coverage_known = 0.0

    # ==============================================
    # C) Unknown(injection) 去向统计 + F1
    # ==============================================
    print("\n===== C) Unknown(injection) 评估 =====")
    # 筛选injection样本
    mask_injection = type_test == 'injection'
    total_injection = int(mask_injection.sum())

    if total_injection > 0:
        # 预测概率最大值
        injection_maxp = np.max(y_proba[mask_injection], axis=1)
        # 预测标签
        injection_pred = y_pred[mask_injection]

        # 预测为Unknown的样本 (概率低于阈值)
        pred_unknown = int((injection_maxp < unknown_threshold).sum())

        # 【核心修正】调整括号位置：先做按位与，再求和
        # 正确逻辑：同时满足「预测为normal」和「概率≥阈值」的样本数量
        pred_normal = int(((injection_pred == normal_idx) & (injection_maxp >= unknown_threshold)).sum())

        # 预测为已知攻击的样本
        pred_known_attack = total_injection - pred_unknown - pred_normal

        # Unknown检测F1分数
        # 真实标签: injection=1, 其他=0
        y_unknown_true = mask_injection.astype(int)
        # 预测标签: 判定为Unknown=1, 其他=0
        y_unknown_pred = np.where((np.max(y_proba, axis=1) < unknown_threshold), 1, 0)

        unknown_precision = precision_score(y_unknown_true, y_unknown_pred, zero_division=0)
        unknown_recall = recall_score(y_unknown_true, y_unknown_pred, zero_division=0)
        unknown_f1 = f1_score(y_unknown_true, y_unknown_pred, zero_division=0)

        # 输出统计结果
        print(f"Injection总样本数: {total_injection}")
        print(f"  预测为Normal: {pred_normal} ({pred_normal / total_injection * 100:.2f}%)")
        print(f"  预测为已知攻击: {pred_known_attack} ({pred_known_attack / total_injection * 100:.2f}%)")
        print(f"  预测为Unknown: {pred_unknown} ({pred_unknown / total_injection * 100:.2f}%)")
        print(f"Unknown检测F1分数: {unknown_f1:.4f}")
        print(f"Unknown检测精确率: {unknown_precision:.4f}")
        print(f"Unknown检测召回率: {unknown_recall:.4f}")
    else:
        print("无injection样本用于评估")
        pred_normal = 0
        pred_known_attack = 0
        pred_unknown = 0
        unknown_f1 = 0.0

    # 保存所有结果
    results = {
        # 二分类结果
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        "binary_auc": binary_auc,
        "confusion_matrix_TN": cm[0, 0],
        "confusion_matrix_FP": cm[0, 1],
        "confusion_matrix_FN": cm[1, 0],
        "confusion_matrix_TP": cm[1, 1],

        # 已知攻击多分类结果
        "macro_f1_known": macro_f1_known,
        "coverage_known(≥0.90)": coverage_known,

        # Unknown评估结果
        "injection_total": total_injection,
        "injection_to_normal": pred_normal,
        "injection_to_known_attack": pred_known_attack,
        "injection_to_unknown": pred_unknown,
        "unknown_f1": unknown_f1,
        "unknown_precision": unknown_precision if total_injection > 0 else 0.0,
        "unknown_recall": unknown_recall if total_injection > 0 else 0.0
    }

    # 保存详细报告
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write("===== 二分类评估 (normal vs 其他) =====\n")
        f.write(f"精确率: {results['binary_precision']:.4f}\n")
        f.write(f"召回率: {results['binary_recall']:.4f}\n")
        f.write(f"F1分数: {results['binary_f1']:.4f}\n")
        f.write(f"AUROC: {results['binary_auc']:.4f}\n")
        f.write("混淆矩阵:\n")
        f.write(f"{cm}\n")
        f.write(f"TN (正常预测为正常): {cm[0, 0]}\n")
        f.write(f"FP (正常预测为异常): {cm[0, 1]}\n")
        f.write(f"FN (异常预测为正常): {cm[1, 0]}\n")
        f.write(f"TP (异常预测为异常): {cm[1, 1]}\n")

        f.write("\n===== 已知攻击多分类评估 =====\n")
        f.write(f"Macro-F1 (已知攻击): {results['macro_f1_known']:.4f}\n")
        f.write(f"覆盖率 (coverage_known, ≥0.90): {results['coverage_known(≥0.90)']:.4f}\n")

        f.write("\n===== Unknown(injection) 评估 =====\n")
        f.write(f"Injection总样本数: {results['injection_total']}\n")
        f.write(f"预测为Normal: {results['injection_to_normal']}\n")
        f.write(f"预测为已知攻击: {results['injection_to_known_attack']}\n")
        f.write(f"预测为Unknown: {results['injection_to_unknown']}\n")
        f.write(f"Unknown检测F1分数: {results['unknown_f1']:.4f}\n")
        f.write(f"Unknown检测精确率: {results['unknown_precision']:.4f}\n")
        f.write(f"Unknown检测召回率: {results['unknown_recall']:.4f}\n")

    # 保存CSV结果
    pd.DataFrame([results]).to_csv(
        os.path.join(RESULTS_DIR, "evaluation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n评估结果已保存至:")
    print(f"- {os.path.join(RESULTS_DIR, 'evaluation_report.txt')}")
    print(f"- {os.path.join(RESULTS_DIR, 'evaluation_summary.csv')}")


if __name__ == "__main__":
    main()