import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List

TRAIN_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\train_no_injection_with_origidx.csv"
TEST_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\test_full_with_injection_with_origidx.csv"
RESULTS_DIR = "D:\\workspace\\TTS-XGB\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def prepare_data() -> Tuple[
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

    # Only apply StandardScaler
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train, type_train), (X_test, y_test, type_test), le


def train_model(X_train: np.ndarray, y_train: np.ndarray, num_classes: int, n_estimators: int = 100,
                max_depth: int = None) -> RandomForestClassifier:
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    # Prepare data
    (X_train, y_train, type_train), (X_test, y_test, type_test), le = prepare_data()
    class_names = le.classes_
    num_classes = len(class_names)
    print(f"\nFeatures: {X_train.shape[1]}, Classes: {num_classes}")

    # Train the Random Forest model
    model = train_model(X_train, y_train, num_classes=num_classes)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # ==============================================
    # A) 二分类评估 (normal算一类，其它算一类)
    # ==============================================
    print("\n===== A) 二分类评估 (normal vs others) =====")
    normal_label = 'normal'
    if normal_label not in class_names:
        raise ValueError(f"Class '{normal_label}' not found in labels")

    normal_idx = le.transform([normal_label])[0]
    y_test_binary = (y_test == normal_idx).astype(int)  # 1 for normal, 0 for others
    y_pred_binary = (y_pred == normal_idx).astype(int)
    y_proba_binary = y_proba[:, normal_idx]  # 正例(normal)的概率

    # 计算二分类指标
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    auroc = roc_auc_score(y_test_binary, y_proba_binary)
    cm = confusion_matrix(y_test_binary, y_pred_binary)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("\n2x2 Confusion Matrix:")
    print("Rows: True Labels (1=normal, 0=others)")
    print("Cols: Predicted Labels (1=normal, 0=others)")
    print(cm)

    # ==============================================
    # B) 已知攻击多分类评估 (除了injection)
    # ==============================================
    print("\n===== B) 已知攻击多分类评估 (excluding injection) =====")
    unknown_label = 'injection'
    if unknown_label not in class_names:
        raise ValueError(f"Class '{unknown_label}' not found in labels")

    # 筛选已知类样本（排除injection）
    known_mask = (type_test != unknown_label)
    known_class_names = [cls for cls in class_names if cls != unknown_label]

    if sum(known_mask) == 0:
        print("No known samples for evaluation")
        macro_f1_known = 0.0
        coverage_known = 0.0
    else:
        # 映射已知类标签到0~N-1范围
        known_le = LabelEncoder()
        known_le.fit(known_class_names)
        y_test_known = known_le.transform(type_test[known_mask])
        y_pred_known = known_le.transform([class_names[p] for p in y_pred[known_mask]])

        # 计算指标
        macro_f1_known = f1_score(y_test_known, y_pred_known, average='macro', zero_division=0)
        coverage_known = sum(known_mask) / len(type_test)  # 已知类占总样本比例

        print(f"Known classes: {known_class_names}")
        print(f"Macro-F1_known: {macro_f1_known:.4f}")
        print(f"coverage_known: {coverage_known:.4f} (≥0.90 target)")

    # ==============================================
    # C) Unknown(injection) 去向统计 + Unknown 精确率/召回率/F1
    # ==============================================
    print("\n===== C) Unknown(injection) Analysis =====")
    # 筛选injection样本（真实Unknown）和非injection样本（真实Known）
    injection_mask = (type_test == unknown_label)
    non_injection_mask = (type_test != unknown_label)
    total_injection = int(sum(injection_mask))
    total_non_injection = int(sum(non_injection_mask))
    print(f"Total injection (Unknown) samples: {total_injection}")
    print(f"Total non-injection (Known) samples: {total_non_injection}")

    # 初始化Unknown指标
    unknown_precision = 0.0
    unknown_recall = 0.0
    unknown_f1 = 0.0
    pred_normal = 0
    pred_unknown = 0
    pred_known_attack = 0

    if total_injection > 0:
        # 预测概率阈值
        unknown_threshold = 0.6
        injection_proba = y_proba[injection_mask]
        non_injection_proba = y_proba[non_injection_mask]
        injection_maxp = np.max(injection_proba, axis=1)
        non_injection_maxp = np.max(non_injection_proba, axis=1)
        injection_pred = y_pred[injection_mask]

        # 去向统计
        pred_normal = sum((injection_pred == normal_idx) & (injection_maxp >= unknown_threshold))
        pred_unknown = sum(injection_maxp < unknown_threshold)
        pred_known_attack = total_injection - pred_normal - pred_unknown

        # 构建全量样本的Unknown检测二值标签（用于计算精确率和召回率）
        # 真实标签: 1=injection(Unknown), 0=non-injection(Known)
        gt_unknown_full = np.zeros(len(type_test))
        gt_unknown_full[injection_mask] = 1

        # 预测标签: 1=判定为Unknown(最大概率<threshold), 0=判定为Known(最大概率≥threshold)
        pred_unknown_full = np.zeros(len(type_test))
        all_maxp = np.max(y_proba, axis=1)
        pred_unknown_full[all_maxp < unknown_threshold] = 1

        # 计算Unknown的精确率、召回率、F1
        unknown_precision = precision_score(gt_unknown_full, pred_unknown_full, zero_division=0)
        unknown_recall = recall_score(gt_unknown_full, pred_unknown_full, zero_division=0)
        unknown_f1 = f1_score(gt_unknown_full, pred_unknown_full, zero_division=0)

        # 输出Unknown指标
        print(f"\nUnknown (injection) Evaluation Metrics:")
        print(f"  Unknown Precision: {unknown_precision:.4f}")
        print(f"  Unknown Recall: {unknown_recall:.4f}")
        print(f"  Unknown F1 Score: {unknown_f1:.4f}")

        # 输出去向统计
        print("\nInjection Routing:")
        pct_normal = (pred_normal / total_injection * 100) if total_injection else 0
        pct_unknown = (pred_unknown / total_injection * 100) if total_injection else 0
        pct_known_attack = (pred_known_attack / total_injection * 100) if total_injection else 0
        print(f"  → Normal: {pred_normal} ({pct_normal:.2f}%)")
        print(f"  → Unknown: {pred_unknown} ({pct_unknown:.2f}%)")
        print(f"  → Known Attack: {pred_known_attack} ({pct_known_attack:.2f}%)")
    else:
        print("No injection samples for evaluation")

    # ==============================================
    # 保存结果
    # ==============================================
    results = {
        # 二分类结果
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_auroc": auroc,
        "confusion_matrix": cm.tolist(),

        # 已知攻击多分类结果
        "macro_f1_known": macro_f1_known,
        "coverage_known": coverage_known,

        # Unknown分析结果
        "total_injection": total_injection,
        "unknown_precision": unknown_precision,
        "unknown_recall": unknown_recall,
        "unknown_f1": unknown_f1,
        "injection_to_normal": pred_normal,
        "injection_to_unknown": pred_unknown,
        "injection_to_known_attack": pred_known_attack
    }

    # 保存为CSV
    results_df = pd.DataFrame([results])
    results_path = os.path.join(RESULTS_DIR, "rf_evaluation_results_with_unknown_pr.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    main()