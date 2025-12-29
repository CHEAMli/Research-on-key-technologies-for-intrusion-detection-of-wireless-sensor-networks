import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ================= 数据加载和预处理函数 =================
def load_and_preprocess_data():
    """加载并预处理数据"""
    data_path = r"D:\workspace\TTS-XGB\data\TON_IoT\Train_Test_datasets\Train_Test_Network_dataset\train_test_network.csv"
    data = pd.read_csv(data_path)

    # 删除非行为特征列
    DROP_COLS = [
        "src_ip", "dst_ip", "proto", "service", "conn_state",
        "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
        "http_method", "http_uri", "http_version",
        "weird_name", "weird_addl", "weird_notice", "label"
    ]
    DROP_COLS = [c for c in DROP_COLS if c in data.columns]
    data = data.drop(columns=DROP_COLS)

    # 处理缺失值
    data = data.replace("-", np.nan)
    for col in data.columns:
        if col != "type":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.fillna(0)

    # 编码标签
    y, class_names = pd.factorize(data["type"])
    X = data.drop(columns=["type"]).values
    num_classes = len(class_names)

    # 按7:1.5:1.5划分训练集、验证集、测试集
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, class_names, num_classes


# ================= 预处理函数 (KMeans+PCA) =================
def kmeans_pca_preprocess(X_train_raw, X_val_raw, X_test_raw, run):
    """使用KMeans聚类特征+PCA降维，仅对训练集拟合"""
    # 1. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)  # 仅用训练集拟合
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 2. KMeans聚类 - 为训练数据添加聚类特征
    n_clusters = min(10, X_train_scaled.shape[0] // 10)  # 聚类数设置
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 + run, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled)  # 仅用训练集拟合

    # 将聚类结果转换为独热编码特征
    train_cluster_features = np.zeros((X_train_scaled.shape[0], n_clusters))
    train_cluster_features[np.arange(X_train_scaled.shape[0]), train_clusters] = 1

    # 为验证集和测试集生成聚类特征（使用训练好的KMeans）
    val_clusters = kmeans.predict(X_val_scaled)
    val_cluster_features = np.zeros((X_val_scaled.shape[0], n_clusters))
    val_cluster_features[np.arange(X_val_scaled.shape[0]), val_clusters] = 1

    test_clusters = kmeans.predict(X_test_scaled)
    test_cluster_features = np.zeros((X_test_scaled.shape[0], n_clusters))
    test_cluster_features[np.arange(X_test_scaled.shape[0]), test_clusters] = 1

    # 3. 合并原始特征与聚类特征
    X_train_combined = np.hstack([X_train_scaled, train_cluster_features])
    X_val_combined = np.hstack([X_val_scaled, val_cluster_features])
    X_test_combined = np.hstack([X_test_scaled, test_cluster_features])

    # 4. PCA降维 - 保留95%的方差
    pca = PCA(n_components=0.95, random_state=42 + run)
    X_train_pca = pca.fit_transform(X_train_combined)  # 仅用训练集拟合
    X_val_pca = pca.transform(X_val_combined)
    X_test_pca = pca.transform(X_test_combined)

    print(f"PCA降维: 原始特征{X_train_combined.shape[1]} -> 降维后{X_train_pca.shape[1]}")

    return X_train_pca, X_val_pca, X_test_pca


# ================= 输出类别指标的文本表格 =================
def print_class_metrics_text_table(class_metrics, class_names, model_name):
    """以文本表格形式输出类别指标"""
    print(f"\n{model_name} 每类别平均指标:")
    print("=" * 80)
    print(f"{'类别':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)

    for idx, cls in enumerate(class_names):
        print(f"{cls:<15} {class_metrics['accuracy'][idx]:<12.4f} "
              f"{class_metrics['precision'][idx]:<12.4f} "
              f"{class_metrics['recall'][idx]:<12.4f} "
              f"{class_metrics['f1'][idx]:<12.4f}")

    print("-" * 80)
    print(f"{'平均':<15} {np.mean(class_metrics['accuracy']):<12.4f} "
          f"{np.mean(class_metrics['precision']):<12.4f} "
          f"{np.mean(class_metrics['recall']):<12.4f} "
          f"{np.mean(class_metrics['f1']):<12.4f}")
    print("=" * 80)


# ================= 主实验函数 =================
def run_kmeans_pca_experiment():
    """运行KMeans+PCA实验"""
    print("=" * 80)
    print("TON-IoT 入侵检测实验 (KMeans+PCA预处理)")
    print("=" * 80)

    # 加载数据
    print("加载数据中...")
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, class_names, num_classes = load_and_preprocess_data()

    print(f"\n发现类别: {list(class_names)}")
    print(f"类别数量: {num_classes}")

    print(f"\n数据集大小:")
    print(f"训练集: {len(X_train_raw)} 样本 ({len(X_train_raw) / len(X_train_raw) * 100:.1f}%)")
    print(
        f"验证集: {len(X_val_raw)} 样本 ({len(X_val_raw) / (len(X_train_raw) + len(X_val_raw) + len(X_test_raw)) * 100:.1f}%)")
    print(
        f"测试集: {len(X_test_raw)} 样本 ({len(X_test_raw) / (len(X_train_raw) + len(X_val_raw) + len(X_test_raw)) * 100:.1f}%)")

    # 存储5次实验的结果
    rf_results = []
    xgb_results = []
    last_rf_cm = None
    last_xgb_cm = None
    last_y_test = None
    last_rf_pred = None
    last_xgb_pred = None

    print(f"\n开始进行5次重复实验...")

    for run in range(5):
        print(f"\n{'=' * 40}")
        print(f"实验轮次 {run + 1}/5")
        print(f"{'=' * 40}")

        # 使用KMeans+PCA预处理（仅训练集拟合）
        X_train, X_val, X_test = kmeans_pca_preprocess(X_train_raw, X_val_raw, X_test_raw, run)

        # 初始化模型
        rf_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42 + run,
            n_jobs=-1
        )

        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42 + run,
            n_jobs=-1
        )

        # 训练和评估RF模型
        print(f"\n训练Random Forest模型...")
        rf_model.fit(X_train, y_train)
        y_pred_val_rf = rf_model.predict(X_val)
        val_accuracy_rf = accuracy_score(y_val, y_pred_val_rf)
        print(f"RF验证集准确率: {val_accuracy_rf:.4f}")

        y_pred_rf = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_precision = precision_score(y_test, y_pred_rf, average="macro", zero_division=0)
        rf_recall = recall_score(y_test, y_pred_rf, average="macro", zero_division=0)
        rf_macro_f1 = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)

        # 计算每个类别的指标
        rf_class_precision = precision_score(y_test, y_pred_rf, average=None, zero_division=0)
        rf_class_recall = recall_score(y_test, y_pred_rf, average=None, zero_division=0)
        rf_class_f1 = f1_score(y_test, y_pred_rf, average=None, zero_division=0)
        rf_class_accuracy = []
        for cls_idx in range(num_classes):
            cls_mask = (y_test == cls_idx)
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred_rf[cls_mask]) if np.sum(cls_mask) > 0 else 0.0
            rf_class_accuracy.append(cls_accuracy)

        print(
            f"Random Forest测试集 | Acc={rf_accuracy:.4f}, Prec={rf_precision:.4f}, Rec={rf_recall:.4f}, Macro-F1={rf_macro_f1:.4f}")

        # 训练和评估XGB模型
        print(f"\n训练XGBoost模型...")
        xgb_model.fit(X_train, y_train)
        y_pred_val_xgb = xgb_model.predict(X_val)
        val_accuracy_xgb = accuracy_score(y_val, y_pred_val_xgb)
        print(f"XGB验证集准确率: {val_accuracy_xgb:.4f}")

        y_pred_xgb = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
        xgb_precision = precision_score(y_test, y_pred_xgb, average="macro", zero_division=0)
        xgb_recall = recall_score(y_test, y_pred_xgb, average="macro", zero_division=0)
        xgb_macro_f1 = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)

        # 计算每个类别的指标
        xgb_class_precision = precision_score(y_test, y_pred_xgb, average=None, zero_division=0)
        xgb_class_recall = recall_score(y_test, y_pred_xgb, average=None, zero_division=0)
        xgb_class_f1 = f1_score(y_test, y_pred_xgb, average=None, zero_division=0)
        xgb_class_accuracy = []
        for cls_idx in range(num_classes):
            cls_mask = (y_test == cls_idx)
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred_xgb[cls_mask]) if np.sum(cls_mask) > 0 else 0.0
            xgb_class_accuracy.append(cls_accuracy)

        print(
            f"XGBoost测试集 | Acc={xgb_accuracy:.4f}, Prec={xgb_precision:.4f}, Rec={xgb_recall:.4f}, Macro-F1={xgb_macro_f1:.4f}")

        # 存储结果
        rf_results.append({
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'macro_f1': rf_macro_f1,
            'class_accuracy': rf_class_accuracy,
            'class_precision': rf_class_precision,
            'class_recall': rf_class_recall,
            'class_f1': rf_class_f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
        })

        xgb_results.append({
            'accuracy': xgb_accuracy,
            'precision': xgb_precision,
            'recall': xgb_recall,
            'macro_f1': xgb_macro_f1,
            'class_accuracy': xgb_class_accuracy,
            'class_precision': xgb_class_precision,
            'class_recall': xgb_class_recall,
            'class_f1': xgb_class_f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred_xgb)
        })

        # 保存最后一次的结果
        if run == 4:
            last_rf_cm = confusion_matrix(y_test, y_pred_rf)
            last_xgb_cm = confusion_matrix(y_test, y_pred_xgb)
            last_y_test = y_test
            last_rf_pred = y_pred_rf
            last_xgb_pred = y_pred_xgb

    # 计算平均结果
    print(f"\n{'=' * 80}")
    print("5次重复实验平均结果")
    print(f"{'=' * 80}")

    # RF平均结果
    rf_avg_accuracy = np.mean([r['accuracy'] for r in rf_results])
    rf_avg_precision = np.mean([r['precision'] for r in rf_results])
    rf_avg_recall = np.mean([r['recall'] for r in rf_results])
    rf_avg_macro_f1 = np.mean([r['macro_f1'] for r in rf_results])

    # XGB平均结果
    xgb_avg_accuracy = np.mean([r['accuracy'] for r in xgb_results])
    xgb_avg_precision = np.mean([r['precision'] for r in xgb_results])
    xgb_avg_recall = np.mean([r['recall'] for r in xgb_results])
    xgb_avg_macro_f1 = np.mean([r['macro_f1'] for r in xgb_results])

    # 计算每个类别的平均指标
    rf_class_accuracy = np.mean([r['class_accuracy'] for r in rf_results], axis=0)
    rf_class_precision = np.mean([r['class_precision'] for r in rf_results], axis=0)
    rf_class_recall = np.mean([r['class_recall'] for r in rf_results], axis=0)
    rf_class_f1 = np.mean([r['class_f1'] for r in rf_results], axis=0)

    xgb_class_accuracy = np.mean([r['class_accuracy'] for r in xgb_results], axis=0)
    xgb_class_precision = np.mean([r['class_precision'] for r in xgb_results], axis=0)
    xgb_class_recall = np.mean([r['class_recall'] for r in xgb_results], axis=0)
    xgb_class_f1 = np.mean([r['class_f1'] for r in xgb_results], axis=0)

    # 输出RF结果
    print("\n" + "=" * 80)
    print("Random Forest 结果")
    print("=" * 80)
    print(f"\nRF平均指标:")
    print(f"Accuracy:  {rf_avg_accuracy:.4f}")
    print(f"Precision: {rf_avg_precision:.4f}")
    print(f"Recall:    {rf_avg_recall:.4f}")
    print(f"Macro-F1:  {rf_avg_macro_f1:.4f}")
    rf_class_metrics = {
        'accuracy': rf_class_accuracy,
        'precision': rf_class_precision,
        'recall': rf_class_recall,
        'f1': rf_class_f1
    }
    print_class_metrics_text_table(rf_class_metrics, class_names, "Random Forest")

    # 输出XGB结果
    print("\n" + "=" * 80)
    print("XGBoost 结果")
    print("=" * 80)
    print(f"\nXGB平均指标:")
    print(f"Accuracy:  {xgb_avg_accuracy:.4f}")
    print(f"Precision: {xgb_avg_precision:.4f}")
    print(f"Recall:    {xgb_avg_recall:.4f}")
    print(f"Macro-F1:  {xgb_avg_macro_f1:.4f}")
    xgb_class_metrics = {
        'accuracy': xgb_class_accuracy,
        'precision': xgb_class_precision,
        'recall': xgb_class_recall,
        'f1': xgb_class_f1
    }
    print_class_metrics_text_table(xgb_class_metrics, class_names, "XGBoost")

    # 输出整体对比
    print("\n" + "=" * 60)
    print("模型整体性能对比")
    print("=" * 60)
    print(f"{'模型':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Macro-F1':<12}")
    print("-" * 60)
    print(f"{'Random Forest':<15} {rf_avg_accuracy:<12.4f} "
          f"{rf_avg_precision:<12.4f} {rf_avg_recall:<12.4f} "
          f"{rf_avg_macro_f1:<12.4f}")
    print(f"{'XGBoost':<15} {xgb_avg_accuracy:<12.4f} "
          f"{xgb_avg_precision:<12.4f} {xgb_avg_recall:<12.4f} "
          f"{xgb_avg_macro_f1:<12.4f}")
    print("=" * 60)

    # 输出最后一次实验的混淆矩阵和分类报告
    print("\n" + "=" * 80)
    print("最后一次实验的混淆矩阵")
    print("=" * 80)
    print(f"\nRandom Forest 混淆矩阵:")
    print("标签顺序:", list(class_names))
    print("\n" + "-" * 70)
    cm_df_rf = pd.DataFrame(last_rf_cm,
                            index=[f"True_{cls}" for cls in class_names],
                            columns=[f"Pred_{cls}" for cls in class_names])
    print(cm_df_rf)

    print(f"\nXGBoost 混淆矩阵:")
    print("标签顺序:", list(class_names))
    print("\n" + "-" * 70)
    cm_df_xgb = pd.DataFrame(last_xgb_cm,
                             index=[f"True_{cls}" for cls in class_names],
                             columns=[f"Pred_{cls}" for cls in class_names])
    print(cm_df_xgb)

    print("\n" + "=" * 80)
    print("最后一次实验的详细分类报告")
    print("=" * 80)
    print(f"\nRandom Forest 分类报告:")
    print(classification_report(last_y_test, last_rf_pred,
                                target_names=class_names, digits=4))
    print(f"\nXGBoost 分类报告:")
    print(classification_report(last_y_test, last_xgb_pred,
                                target_names=class_names, digits=4))

    return {
        'rf_avg': {
            'accuracy': rf_avg_accuracy,
            'precision': rf_avg_precision,
            'recall': rf_avg_recall,
            'macro_f1': rf_avg_macro_f1,
            'class_accuracy': rf_class_accuracy,
            'class_precision': rf_class_precision,
            'class_recall': rf_class_recall,
            'class_f1': rf_class_f1
        },
        'xgb_avg': {
            'accuracy': xgb_avg_accuracy,
            'precision': xgb_avg_precision,
            'recall': xgb_avg_recall,
            'macro_f1': xgb_avg_macro_f1,
            'class_accuracy': xgb_class_accuracy,
            'class_precision': xgb_class_precision,
            'class_recall': xgb_class_recall,
            'class_f1': xgb_class_f1
        }
    }


# ================= 运行实验 =================
if __name__ == "__main__":
    try:
        run_kmeans_pca_experiment()
        print("\n" + "=" * 80)
        print("实验完成!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        print("请检查文件路径是否正确:")
        print("D:/workspace/TTS-XGB/data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback

        traceback.print_exc()