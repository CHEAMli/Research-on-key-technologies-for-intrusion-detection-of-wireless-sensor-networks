import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, Dict, List

# 路径配置
TRAIN_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\train_no_injection_with_origidx.csv"
TEST_FILE = "D:\\workspace\\TTS-XGB\\data\\new TON-IoT\\test_full_with_injection_with_origidx.csv"
RESULTS_DIR = "D:\\workspace\\TTS-XGB\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_global_label_mapping(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """创建全局标签映射（解决训练集类别不全问题）"""
    # 1. 收集训练+测试集的所有唯一标签
    train_labels = train_df['type'].unique()
    test_labels = test_df['type'].unique()
    all_labels = sorted(list(set(train_labels) | set(test_labels)))  # 合并并排序

    # 2. 建立全局连续索引映射（0开始）
    label_mapping = {label: idx for idx, label in enumerate(all_labels)}

    # 打印类别分布信息
    print("\n===== 类别分布详情 =====")
    print(f"所有类别（共{len(all_labels)}个）: {all_labels}")
    print(f"训练集包含类别: {sorted(train_labels)}")
    print(f"测试集包含类别: {sorted(test_labels)}")
    print(f"训练集缺失的类别: {[l for l in all_labels if l not in train_labels]}")
    print(f"全局标签映射: {label_mapping}")

    return label_mapping, all_labels


def prepare_data(n_clusters: int = 10, pca_components: int = 10) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Dict[str, int], List[str]
]:
    """数据预处理"""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 数值特征列表
    numeric_features = ['src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
                        'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
                        'dns_qclass', 'dns_qtype', 'dns_rcode',
                        'http_request_body_len', 'http_response_body_len', 'http_status_code']

    # 创建全局标签映射
    label_mapping, all_labels = create_global_label_mapping(train_df, test_df)

    # 转换标签为连续索引
    y_train = np.array([label_mapping[cls] for cls in train_df['type'].values])
    y_test = np.array([label_mapping[cls] for cls in test_df['type'].values])

    # 原始类型保留
    type_train = train_df['type'].values
    type_test = test_df['type'].values

    # 特征预处理
    X_train = train_df[numeric_features].fillna(0).values.astype(np.float32)
    X_test = test_df[numeric_features].fillna(0).values.astype(np.float32)

    # Step 1: 标准化
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 2: KMeans聚类
    print(f"Applying KMeans clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)

    # 增加聚类特征
    X_train_with_cluster = np.column_stack([X_train, train_clusters])
    X_test_with_cluster = np.column_stack([X_test, test_clusters])

    # Step 3: PCA降维
    print(f"Applying PCA (n_components={pca_components})...")
    pca = PCA(n_components=pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_with_cluster)
    X_test_pca = pca.transform(X_test_with_cluster)

    print(f"PCA解释方差占比: {pca.explained_variance_ratio_.sum():.4f}")

    return (X_train_pca, y_train, type_train), (X_test_pca, y_test, type_test), label_mapping, all_labels


def train_xgb_native(X_train: np.ndarray, y_train: np.ndarray, num_classes: int,
                     n_estimators: int = 100, max_depth: int = 3) -> xgb.Booster:
    """
    使用XGBoost原生接口训练模型（彻底绕开sklearn的类别校验）
    """
    print("\nTraining XGBoost model (native API)...")

    # 1. 转换为XGBoost的DMatrix格式（原生接口专用）
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 2. 设置XGBoost参数（无类别校验）
    params = {
        'objective': 'multi:softprob',  # 输出概率矩阵
        'num_class': num_classes,  # 全局类别总数
        'max_depth': max_depth,  # 树深度
        'eta': 0.1,  # 学习率（对应sklearn的learning_rate）
        'seed': 42,  # 随机种子
        'silent': 1,  # 静默模式
        'eval_metric': 'mlogloss'  # 多分类损失
    }

    # 3. 训练模型（原生train接口，无任何类别校验）
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,  # 对应sklearn的n_estimators
        verbose_eval=False  # 不打印训练日志
    )

    return booster


def predict_xgb_native(booster: xgb.Booster, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用原生接口预测：返回类别索引 + 概率矩阵
    """
    dtest = xgb.DMatrix(X)
    prob_matrix = booster.predict(dtest)  # 输出形状: (n_samples, num_classes)
    y_pred = np.argmax(prob_matrix, axis=1)  # 取概率最大的类别
    return y_pred, prob_matrix


def main():
    # 超参数
    n_clusters = int(os.environ.get("N_CLUSTERS", 10))
    pca_components = int(os.environ.get("PCA_COMPONENTS", 10))
    n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
    max_depth = int(os.environ.get("MAX_DEPTH", 3))
    unknown_threshold = 0.6  # Unknown判断阈值

    # 数据预处理
    (X_train, y_train, type_train), (X_test, y_test, type_test), label_mapping, all_labels = prepare_data(
        n_clusters=n_clusters,
        pca_components=pca_components
    )

    # 全局类别总数
    num_classes = len(all_labels)
    print(f"\n特征维度: {X_train.shape[1]}, 全局类别总数: {num_classes}")

    # 获取关键标签索引
    try:
        normal_idx = label_mapping['normal']
        injection_idx = label_mapping['injection']
    except KeyError as e:
        print(f"错误：未找到关键标签 {e}")
        return

    # 训练模型（原生接口，无类别校验）
    booster = train_xgb_native(
        X_train,
        y_train,
        num_classes=num_classes,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # 预测（原生接口）
    print("Predicting on test set...")
    y_pred, prob_matrix = predict_xgb_native(booster, X_test)
    y_max_prob = np.max(prob_matrix, axis=1)  # 每个样本的最大概率

    # 应用Unknown检测（概率低于阈值的视为Unknown）
    final_pred = np.where(y_max_prob < unknown_threshold, -1, y_pred)  # -1表示Unknown

    # ============== A) 二分类评估（normal vs 其他） ==============
    print("\n===== A) 二分类评估 (normal vs 其他) =====")
    # 真实标签：normal=0，其他=1
    y_binary_true = np.where(type_test == 'normal', 0, 1)
    # 预测标签：normal=0，其他=1（包括Unknown）
    y_binary_pred = np.where((final_pred == normal_idx) & (y_max_prob >= unknown_threshold), 0, 1)

    # 计算指标
    binary_precision = precision_score(y_binary_true, y_binary_pred, zero_division=0)
    binary_recall = recall_score(y_binary_true, y_binary_pred, zero_division=0)
    binary_f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)

    # 计算AUROC（使用normal类的概率作为评分）
    normal_prob = prob_matrix[:, normal_idx] if normal_idx < prob_matrix.shape[1] else np.zeros(len(y_test))
    binary_auc = roc_auc_score(y_binary_true, 1 - normal_prob)  # 1-normal概率作为异常分数

    # 混淆矩阵
    cm = confusion_matrix(y_binary_true, y_binary_pred)

    # 打印结果
    print(f"精确率 (Precision): {binary_precision:.4f}")
    print(f"召回率 (Recall): {binary_recall:.4f}")
    print(f"F1分数: {binary_f1:.4f}")
    print(f"AUROC: {binary_auc:.4f}")
    print("混淆矩阵:")
    print(cm)
    print("混淆矩阵解释:")
    print(f"TN: {cm[0, 0]} (正常被正确预测为正常)")
    print(f"FP: {cm[0, 1]} (正常被预测为异常)")
    print(f"FN: {cm[1, 0]} (异常被预测为正常)")
    print(f"TP: {cm[1, 1]} (异常被正确预测为异常)")

    # ============== B) 已知攻击多分类评估（除injection） ==============
    print("\n===== B) 已知攻击多分类评估 (除injection) =====")
    # 筛选已知类样本（非injection且非Unknown）
    known_mask = (type_test != 'injection') & (final_pred != -1)
    known_count = np.sum(known_mask)
    total_known_samples = np.sum(type_test != 'injection')

    if known_count > 0:
        # 已知类真实标签和预测标签
        y_known_true = y_test[known_mask]
        y_known_pred = final_pred[known_mask]

        # 过滤掉injection相关标签
        valid_mask = ~np.isin(y_known_true, injection_idx)
        y_known_true = y_known_true[valid_mask]
        y_known_pred = y_known_pred[valid_mask]

        if len(y_known_true) > 0:
            macro_f1_known = f1_score(y_known_true, y_known_pred, average='macro', zero_division=0)
            coverage_known = known_count / total_known_samples if total_known_samples > 0 else 0.0

            print(f"Macro-F1 (已知类): {macro_f1_known:.4f}")
            print(f"覆盖率 (Coverage): {coverage_known:.4f} (≥0.90为达标)")
        else:
            print("无有效已知类样本用于评估")
    else:
        print("无已知类样本用于评估")

    # ============== C) Unknown(injection) 去向统计 + Unknown F1 ==============
    print("\n===== C) Unknown(injection) 分析 =====")
    # 筛选injection样本
    injection_mask = (type_test == 'injection')
    total_injection = np.sum(injection_mask)

    if total_injection == 0:
        print("测试集中未找到injection样本")
    else:
        # injection样本的预测结果
        injection_pred = final_pred[injection_mask]
        injection_max_prob = y_max_prob[injection_mask]

        # 统计去向
        pred_normal = np.sum((injection_pred == normal_idx) & (injection_max_prob >= unknown_threshold))
        pred_unknown = np.sum(injection_pred == -1)  # 被识别为Unknown
        pred_other_attack = total_injection - pred_normal - pred_unknown  # 被识别为其他已知攻击

        # 计算Unknown检测的F1分数
        # 真实标签：injection=1（视为Unknown），其他=0
        y_unknown_true = np.where(type_test == 'injection', 1, 0)
        # 预测标签：被识别为Unknown=1，其他=0
        y_unknown_pred = np.where(final_pred == -1, 1, 0)

        unknown_precision = precision_score(y_unknown_true, y_unknown_pred, zero_division=0)
        unknown_recall = recall_score(y_unknown_true, y_unknown_pred, zero_division=0)
        unknown_f1 = f1_score(y_unknown_true, y_unknown_pred, zero_division=0)

        # 打印结果
        print(f"Injection样本总数: {total_injection}")
        print(f"  → 被预测为Normal: {pred_normal} ({pred_normal / total_injection * 100:.2f}%)")
        print(f"  → 被预测为Unknown: {pred_unknown} ({pred_unknown / total_injection * 100:.2f}%)")
        print(f"  → 被预测为其他攻击: {pred_other_attack} ({pred_other_attack / total_injection * 100:.2f}%)")
        print(f"\nUnknown检测F1分数: {unknown_f1:.4f}")
        print(f"Unknown检测精确率: {unknown_precision:.4f}")
        print(f"Unknown检测召回率: {unknown_recall:.4f}")

    # 保存结果
    print("\n📁 结果文件保存路径:")
    report_path = os.path.join(RESULTS_DIR, "xgb_model_complete_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== 完整评估报告 =====\n\n")

        # 保存二分类结果
        f.write("A) 二分类评估 (normal vs 其他)\n")
        f.write(f"精确率: {binary_precision:.4f}\n")
        f.write(f"召回率: {binary_recall:.4f}\n")
        f.write(f"F1分数: {binary_f1:.4f}\n")
        f.write(f"AUROC: {binary_auc:.4f}\n")
        f.write("混淆矩阵:\n")
        f.write(f"{cm}\n\n")

        # 保存已知类多分类结果
        f.write("B) 已知攻击多分类评估 (除injection)\n")
        if known_count > 0 and len(y_known_true) > 0:
            f.write(f"Macro-F1 (已知类): {macro_f1_known:.4f}\n")
            f.write(f"覆盖率: {coverage_known:.4f}\n\n")
        else:
            f.write("无有效评估数据\n\n")

        # 保存Unknown分析结果
        f.write("C) Unknown(injection) 分析\n")
        if total_injection > 0:
            f.write(f"Injection样本总数: {total_injection}\n")
            f.write(f"→ Normal: {pred_normal} ({pred_normal / total_injection * 100:.2f}%)\n")
            f.write(f"→ Unknown: {pred_unknown} ({pred_unknown / total_injection * 100:.2f}%)\n")
            f.write(f"→ 其他攻击: {pred_other_attack} ({pred_other_attack / total_injection * 100:.2f}%)\n")
            f.write(f"Unknown F1分数: {unknown_f1:.4f}\n")
        else:
            f.write("测试集中未找到injection样本\n")

    print(f"- 完整评估报告: {report_path}")

    # 保存模型
    model_path = os.path.join(RESULTS_DIR, "xgb_model.model")
    booster.save_model(model_path)
    print(f"- 模型文件: {model_path}")


if __name__ == "__main__":
    main()