import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
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


def prepare_data() -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Dict[str, int], List[str]
]:
    """数据预处理（仅保留标准化）"""
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

    # 特征预处理（仅保留标准化）
    X_train = train_df[numeric_features].fillna(0).values.astype(np.float32)
    X_test = test_df[numeric_features].fillna(0).values.astype(np.float32)

    # 标准化
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"标准化后特征维度: {X_train.shape[1]}")

    return (X_train, y_train, type_train), (X_test, y_test, type_test), label_mapping, all_labels


def train_xgb_native(X_train: np.ndarray, y_train: np.ndarray, num_classes: int,
                     n_estimators: int = 100, max_depth: int = 3) -> xgb.Booster:
    """使用XGBoost原生接口训练模型"""
    print("\nTraining XGBoost model (native API)...")

    # 转换为XGBoost的DMatrix格式
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # 设置XGBoost参数
    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': max_depth,
        'eta': 0.1,
        'seed': 42,
        'silent': 1,
        'eval_metric': 'mlogloss'
    }

    # 训练模型
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False
    )

    return booster


def predict_xgb_native(booster: xgb.Booster, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用原生接口预测：返回类别索引 + 概率矩阵"""
    dtest = xgb.DMatrix(X)
    prob_matrix = booster.predict(dtest)  # 形状: (n_samples, num_classes)
    y_pred = np.argmax(prob_matrix, axis=1)  # 取概率最大的类别
    return y_pred, prob_matrix


def main():
    # 超参数
    n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
    max_depth = int(os.environ.get("MAX_DEPTH", 3))
    unknown_threshold = 0.6  # Unknown检测阈值

    # 数据预处理
    (X_train, y_train, type_train), (X_test, y_test, type_test), label_mapping, all_labels = prepare_data()
    num_classes = len(all_labels)
    print(f"\n特征维度: {X_train.shape[1]}, 全局类别总数: {num_classes}")

    # 训练模型
    booster = train_xgb_native(
        X_train,
        y_train,
        num_classes=num_classes,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    # 预测
    print("Predicting on test set...")
    y_pred, prob_matrix = predict_xgb_native(booster, X_test)
    max_probs = np.max(prob_matrix, axis=1)  # 每个样本的最大概率

    # ============== A) 二分类评估 (normal vs 其他) ==============
    print("\n===== A) 二分类评估 (normal vs 其他) =====")
    try:
        normal_idx = label_mapping['normal']
        # 构建二分类标签：0=normal，1=其他
        y_test_binary = (y_test != normal_idx).astype(int)
        y_pred_binary = (y_pred != normal_idx).astype(int)

        # 计算指标
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

        # 计算AUROC（使用normal类的概率作为评分）
        normal_probs = prob_matrix[:, normal_idx]
        attack_probs = 1 - normal_probs  # 其他类的概率
        auroc = roc_auc_score(y_test_binary, attack_probs)

        # 混淆矩阵
        cm = confusion_matrix(y_test_binary, y_pred_binary)

        # 输出结果
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUROC: {auroc:.4f}")
        print("混淆矩阵:")
        print(cm)
        print("行: 真实标签 (0=normal, 1=其他), 列: 预测标签")

    except KeyError:
        print("⚠️  未找到'normal'标签，无法进行二分类评估")

    # ============== B) 已知攻击多分类评估 (除injection) ==============
    print("\n===== B) 已知攻击多分类评估 (除injection) =====")
    try:
        # 确定已知类别（所有类别除了injection）
        known_labels = [label for label in all_labels if label != 'injection']
        known_indices = [label_mapping[label] for label in known_labels]

        # 筛选已知类样本（测试集中非injection样本）
        mask_known = (type_test != 'injection')
        y_test_known = y_test[mask_known]
        y_pred_known = y_pred[mask_known]
        max_probs_known = max_probs[mask_known]

        if len(y_test_known) == 0:
            print("⚠️  测试集中没有已知类样本")
        else:
            # 计算Macro-F1
            macro_f1_known = f1_score(
                y_test_known, y_pred_known,
                average='macro',
                labels=known_indices,
                zero_division=0
            )

            # 计算覆盖率（预测为已知类且概率≥阈值的样本比例）
            mask_confident = (max_probs_known >= unknown_threshold)
            coverage_known = mask_confident.mean()

            print(f"已知类列表: {known_labels}")
            print(f"Macro-F1_known: {macro_f1_known:.4f}")
            print(f"coverage_known: {coverage_known:.4f} (阈值={unknown_threshold})")

    except Exception as e:
        print(f"⚠️  已知类评估出错: {str(e)}")

    # ============== C) Unknown(injection) 分析 ==============
    print("\n===== C) Unknown(injection) 分析 =====")
    try:
        # 筛选injection样本
        mask_injection = (type_test == 'injection')
        total_injection = int(mask_injection.sum())

        if total_injection == 0:
            print("⚠️  测试集中未找到injection样本")
        else:
            # 获取injection样本的预测结果
            injection_pred = y_pred[mask_injection]
            injection_probs = prob_matrix[mask_injection]
            injection_maxp = max_probs[mask_injection]
            normal_idx = label_mapping['normal']

            # 1. 去向统计
            # 预测为Normal
            pred_normal = int(((injection_pred == normal_idx) & (injection_maxp >= unknown_threshold)).sum())
            # 预测为Unknown
            pred_unknown = int((injection_maxp < unknown_threshold).sum())
            # 预测为其他已知攻击
            pred_known_attack = 0
            known_attack_details = {}
            for label, idx in label_mapping.items():
                if label not in ['injection', 'normal']:
                    count = int(((injection_pred == idx) & (injection_maxp >= unknown_threshold)).sum())
                    pred_known_attack += count
                    if count > 0:
                        known_attack_details[label] = count

            # 2. Unknown检测F1（视injection为正样本，其他为负样本）
            # 构建标签：1=injection，0=其他
            y_unknown_gt = mask_injection.astype(int)
            # 构建预测：1=预测为Unknown，0=其他
            y_unknown_pred = (max_probs < unknown_threshold).astype(int)

            u_precision = precision_score(y_unknown_gt, y_unknown_pred, zero_division=0)
            u_recall = recall_score(y_unknown_gt, y_unknown_pred, zero_division=0)
            u_f1 = f1_score(y_unknown_gt, y_unknown_pred, zero_division=0)

            # 输出结果
            print(f"Injection样本总数: {total_injection}")
            print(f"  → Normal   : {pred_normal} ({pred_normal / total_injection * 100:.2f}%)")
            print(f"  → Unknown  : {pred_unknown} ({pred_unknown / total_injection * 100:.2f}%)")
            print(f"  → 其他已知攻击: {pred_known_attack} ({pred_known_attack / total_injection * 100:.2f}%)")
            if known_attack_details:
                print(f"  详细分布: {known_attack_details}")

            print(f"\nUnknown检测F1: {u_f1:.4f}")
            print(f"Unknown检测精确率: {u_precision:.4f}")
            print(f"Unknown检测召回率: {u_recall:.4f}")

    except KeyError as e:
        print(f"⚠️  未找到标签: {e}")
        print(f"当前全局标签列表: {all_labels}")

    # 保存结果
    print("\n📁 结果文件保存路径:")
    report_path = os.path.join(RESULTS_DIR, "xgb_model_evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== 模型评估报告 =====\n")
        f.write(f"时间: {pd.Timestamp.now()}\n")
        f.write(f"参数: n_estimators={n_estimators}, max_depth={max_depth}, unknown_threshold={unknown_threshold}\n\n")

        # 保存A部分
        f.write("===== A) 二分类评估 (normal vs 其他) =====\n")
        if 'normal_idx' in locals():
            f.write(f"精确率: {precision:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n")
            f.write(f"AUROC: {auroc:.4f}\n")
            f.write("混淆矩阵:\n")
            f.write(f"{cm}\n\n")
        else:
            f.write("未找到'normal'标签，无法评估\n\n")

        # 保存B部分
        f.write("===== B) 已知攻击多分类评估 =====\n")
        if 'macro_f1_known' in locals():
            f.write(f"已知类列表: {known_labels}\n")
            f.write(f"Macro-F1_known: {macro_f1_known:.4f}\n")
            f.write(f"coverage_known: {coverage_known:.4f}\n\n")
        else:
            f.write("已知类评估失败\n\n")

        # 保存C部分
        f.write("===== C) Unknown(injection) 分析 =====\n")
        if total_injection > 0:
            f.write(f"Injection样本总数: {total_injection}\n")
            f.write(f"→ Normal: {pred_normal} ({pred_normal / total_injection * 100:.2f}%)\n")
            f.write(f"→ Unknown: {pred_unknown} ({pred_unknown / total_injection * 100:.2f}%)\n")
            f.write(f"→ 其他已知攻击: {pred_known_attack} ({pred_known_attack / total_injection * 100:.2f}%)\n")
            if known_attack_details:
                f.write(f"详细分布: {known_attack_details}\n")
            f.write(f"Unknown检测F1: {u_f1:.4f}\n")
            f.write(f"Unknown检测精确率: {u_precision:.4f}\n")
            f.write(f"Unknown检测召回率: {u_recall:.4f}\n")
        else:
            f.write("未找到injection样本\n")

    print(f"- 评估报告: {report_path}")
    booster.save_model(os.path.join(RESULTS_DIR, "xgb_model.model"))
    print(f"- 模型文件: {os.path.join(RESULTS_DIR, 'xgb_model.model')}")


if __name__ == "__main__":
    main()