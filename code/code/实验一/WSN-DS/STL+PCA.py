import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings('ignore')
import time

# 设置随机种子保证可复现性
np.random.seed(42)


# 1. 数据加载和预处理
def load_and_preprocess_data():
    """加载数据并按时间顺序划分"""
    print("正在加载数据...")

    # 加载训练集和测试集
    train_df = pd.read_parquet('D:/workspace/TTS-XGB/data/WSN-DS-main/train.parquet')
    test_df = pd.read_parquet('D:/workspace/TTS-XGB/data/WSN-DS-main/test.parquet')

    print(f"训练集原始形状: {train_df.shape}")
    print(f"测试集原始形状: {test_df.shape}")

    # 合并数据用于按时间顺序划分
    full_df = pd.concat([train_df, test_df], axis=0)

    # 确保按时间顺序排序（假设数据集已按时间排序）
    full_df = full_df.sort_index()
    full_df = full_df.reset_index(drop=True)

    # 分离特征和标签
    target_col = 'Attack_type'

    if target_col not in full_df.columns:
        print("列名列表:")
        print(full_df.columns.tolist())
        raise ValueError(f"标签列'{target_col}'不存在于数据中")

    X = full_df.drop(columns=[target_col])
    y = full_df[target_col]

    # 获取类别信息
    classes = y.unique()
    print(f"发现类别: {classes}")
    print(f"类别数量: {len(classes)}")

    # 统计类别分布
    class_distribution = y.value_counts()
    print("\n类别分布:")
    for cls, count in class_distribution.items():
        percentage = count / len(y) * 100
        print(f"  {cls}: {count} ({percentage:.2f}%)")

    return X, y, classes


# 2. SMOTE+TomekLink+PCA预处理
def smote_tomek_pca_preprocessing(X_train, X_val, X_test, y_train, n_components=0.95):
    """SMOTE+TomekLink+PCA预处理"""

    print("\n进行SMOTE+TomekLink处理...")
    print(f"重采样前训练集形状: X={X_train.shape}, y={y_train.shape}")
    print("重采样前类别分布:")
    train_dist = y_train.value_counts()
    for cls, count in train_dist.items():
        percentage = count / len(y_train) * 100
        print(f"  {cls}: {count} ({percentage:.2f}%)")

    # SMOTE+TomekLink重采样
    smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

    print(f"\n重采样后训练集形状: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
    print("重采样后类别分布:")
    resampled_dist = pd.Series(y_train_resampled).value_counts()
    for cls, count in resampled_dist.items():
        percentage = count / len(y_train_resampled) * 100
        print(f"  {cls}: {count} ({percentage:.2f}%)")

    print("\n进行标准化处理...")
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"标准化后训练集形状: {X_train_scaled.shape}")
    print(f"标准化后验证集形状: {X_val_scaled.shape}")
    print(f"标准化后测试集形状: {X_test_scaled.shape}")

    print("\n进行PCA降维处理...")
    # PCA降维
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"PCA后训练集形状: {X_train_pca.shape}")
    print(f"PCA后验证集形状: {X_val_pca.shape}")
    print(f"PCA后测试集形状: {X_test_pca.shape}")
    print(f"PCA后特征维度: {X_train_pca.shape[1]}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_val_pca, X_test_pca, y_train_resampled, X_train_scaled.shape[1]


# 3. 按时间顺序划分数据
def temporal_train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """按时间顺序划分训练集、验证集和测试集"""

    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # 按时间顺序划分
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print(f"\n数据划分结果:")
    print(f"训练集大小: {len(X_train)} ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"验证集大小: {len(X_val)} ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"测试集大小: {len(X_test)} ({len(X_test) / total_samples * 100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# 4. 模型训练和评估
def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, classes):
    """训练和评估单个模型"""

    print(f"\n训练{model_name}模型...")
    start_time = time.time()

    # 训练模型
    model.fit(X_train, y_train)

    # 在验证集上预测
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)

    # 在测试集上预测
    y_test_pred = model.predict(X_test)

    # 计算各项指标
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    # 计算每个类别的指标
    class_precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
    class_recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    class_accuracy = []
    for cls in classes:
        cls_mask = (y_test == cls)
        if np.sum(cls_mask) > 0:
            cls_accuracy = accuracy_score(y_test[cls_mask], y_test_pred[cls_mask])
        else:
            cls_accuracy = 0.0
        class_accuracy.append(cls_accuracy)

    training_time = time.time() - start_time
    print(f"{model_name}训练完成，耗时: {training_time:.2f}秒")
    print(f"验证集准确率: {val_accuracy:.4f}, Macro-F1: {val_f1:.4f}")
    print(f"测试集准确率: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'class_accuracy': class_accuracy,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'training_time': training_time
    }


# 5. 输出类别指标的文本表格
def print_class_metrics_text_table(class_metrics, classes, model_name):
    """以文本表格形式输出类别指标"""

    print(f"\n{model_name} 每类别平均指标:")
    print("=" * 60)

    # 表头
    print(f"{'类别':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)

    # 类别行
    for idx, cls in enumerate(classes):
        print(f"{cls:<15} {class_metrics['accuracy'][idx]:<12.4f} "
              f"{class_metrics['precision'][idx]:<12.4f} "
              f"{class_metrics['recall'][idx]:<12.4f} "
              f"{class_metrics['f1'][idx]:<12.4f}")

    # 平均行
    print("-" * 60)
    print(f"{'平均':<15} {np.mean(class_metrics['accuracy']):<12.4f} "
          f"{np.mean(class_metrics['precision']):<12.4f} "
          f"{np.mean(class_metrics['recall']):<12.4f} "
          f"{np.mean(class_metrics['f1']):<12.4f}")
    print("=" * 60)


# 6. 输出整体指标的文本表格
def print_overall_metrics_text_table(rf_metrics, xgb_metrics):
    """以文本表格形式输出整体指标"""

    print(f"\n模型整体性能对比:")
    print("=" * 50)
    print(f"{'模型':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Macro-F1':<12}")
    print("-" * 50)
    print(f"{'Random Forest':<15} {rf_metrics['accuracy']:<12.4f} "
          f"{rf_metrics['precision']:<12.4f} {rf_metrics['recall']:<12.4f} "
          f"{rf_metrics['macro_f1']:<12.4f}")
    print(f"{'XGBoost':<15} {xgb_metrics['accuracy']:<12.4f} "
          f"{xgb_metrics['precision']:<12.4f} {xgb_metrics['recall']:<12.4f} "
          f"{xgb_metrics['macro_f1']:<12.4f}")
    print("=" * 50)


# 7. 主实验函数
def main_experiment():
    """主实验函数"""

    print("=" * 70)
    print("WSN-DS 入侵检测实验 (SMOTE+TomekLink+PCA预处理)")
    print("=" * 70)

    # 1. 加载数据
    X, y, classes = load_and_preprocess_data()

    # 存储5次实验的结果
    rf_results = []
    xgb_results = []

    # 存储最后一次实验的结果
    last_rf_cm = None
    last_xgb_cm = None
    last_rf_pred = None
    last_xgb_pred = None
    last_y_test = None
    original_feature_count = None

    # 5次重复实验
    n_repeats = 5
    print(f"\n开始进行{n_repeats}次重复实验...")

    for i in range(n_repeats):
        print(f"\n{'=' * 40}")
        print(f"实验轮次 {i + 1}/{n_repeats}")
        print(f"{'=' * 40}")

        # 每次实验重新划分数据（按时间顺序）
        X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
            X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # 保存原始特征数量
        if original_feature_count is None:
            original_feature_count = X_train.shape[1]
            print(f"\n原始特征维度: {original_feature_count}")

        # SMOTE+TomekLink+PCA预处理
        X_train_processed, X_val_processed, X_test_processed, y_train_resampled, standardized_feature_count = smote_tomek_pca_preprocessing(
            X_train, X_val, X_test, y_train
        )

        # 初始化模型
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42 + i,
            n_jobs=-1,
            verbose=0
        )

        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42 + i,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # 训练和评估RF模型
        rf_result = train_and_evaluate_model(
            rf_model, "Random Forest",
            X_train_processed, y_train_resampled,
            X_val_processed, y_val,
            X_test_processed, y_test,
            classes
        )
        rf_results.append(rf_result)

        # 训练和评估XGB模型
        xgb_result = train_and_evaluate_model(
            xgb_model, "XGBoost",
            X_train_processed, y_train_resampled,
            X_val_processed, y_val,
            X_test_processed, y_test,
            classes
        )
        xgb_results.append(xgb_result)

        # 保存最后一次的结果
        if i == n_repeats - 1:
            last_rf_cm = rf_result['confusion_matrix']
            last_xgb_cm = xgb_result['confusion_matrix']
            last_rf_pred = rf_result['y_pred']
            last_xgb_pred = xgb_result['y_pred']
            last_y_test = y_test

    # 8. 计算平均结果
    print(f"\n{'=' * 70}")
    print("5次重复实验平均结果")
    print(f"{'=' * 70}")

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

    # 9. 输出预处理信息
    print(f"\n数据预处理信息:")
    print(f"原始特征维度: {original_feature_count}")
    print(f"标准化后特征维度: {standardized_feature_count}")
    print(f"PCA后特征维度: {X_train_processed.shape[1]}")

    # 10. 输出RF结果
    print("\n" + "=" * 70)
    print("Random Forest 结果")
    print("=" * 70)

    print(f"\nRF平均指标:")
    print(f"Accuracy:  {rf_avg_accuracy:.4f}")
    print(f"Precision: {rf_avg_precision:.4f}")
    print(f"Recall:    {rf_avg_recall:.4f}")
    print(f"Macro-F1:  {rf_avg_macro_f1:.4f}")

    # 输出RF类别指标表格
    rf_class_metrics = {
        'accuracy': rf_class_accuracy,
        'precision': rf_class_precision,
        'recall': rf_class_recall,
        'f1': rf_class_f1
    }
    print_class_metrics_text_table(rf_class_metrics, classes, "Random Forest")

    # 11. 输出XGB结果
    print("\n" + "=" * 70)
    print("XGBoost 结果")
    print("=" * 70)

    print(f"\nXGB平均指标:")
    print(f"Accuracy:  {xgb_avg_accuracy:.4f}")
    print(f"Precision: {xgb_avg_precision:.4f}")
    print(f"Recall:    {xgb_avg_recall:.4f}")
    print(f"Macro-F1:  {xgb_avg_macro_f1:.4f}")

    # 输出XGB类别指标表格
    xgb_class_metrics = {
        'accuracy': xgb_class_accuracy,
        'precision': xgb_class_precision,
        'recall': xgb_class_recall,
        'f1': xgb_class_f1
    }
    print_class_metrics_text_table(xgb_class_metrics, classes, "XGBoost")

    # 12. 输出整体对比
    rf_overall_metrics = {
        'accuracy': rf_avg_accuracy,
        'precision': rf_avg_precision,
        'recall': rf_avg_recall,
        'macro_f1': rf_avg_macro_f1
    }

    xgb_overall_metrics = {
        'accuracy': xgb_avg_accuracy,
        'precision': xgb_avg_precision,
        'recall': xgb_avg_recall,
        'macro_f1': xgb_avg_macro_f1
    }

    print_overall_metrics_text_table(rf_overall_metrics, xgb_overall_metrics)

    # 13. 输出最后一次实验的混淆矩阵
    print("\n" + "=" * 70)
    print("最后一次实验的混淆矩阵")
    print("=" * 70)

    print(f"\nRandom Forest 混淆矩阵 (行:真实标签, 列:预测标签):")
    print("标签顺序:", list(classes))
    print("\n" + "-" * 50)
    print("混淆矩阵:")
    print(last_rf_cm)

    print(f"\nXGBoost 混淆矩阵 (行:真实标签, 列:预测标签):")
    print("标签顺序:", list(classes))
    print("\n" + "-" * 50)
    print("混淆矩阵:")
    print(last_xgb_cm)

    # 14. 返回结果
    return {
        'rf_results': rf_results,
        'xgb_results': xgb_results,
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
        },
        'last_confusion_matrices': {
            'rf': last_rf_cm,
            'xgb': last_xgb_cm
        },
        'classes': classes,
        'feature_info': {
            'original': original_feature_count,
            'standardized': standardized_feature_count,
            'pca': X_train_processed.shape[1]
        }
    }


# 运行实验
if __name__ == "__main__":
    try:
        print("开始实验 (SMOTE+TomekLink+PCA预处理)...")
        print("注意: 需要安装imbalanced-learn库，可使用: pip install imbalanced-learn")

        results = main_experiment()

        print("\n" + "=" * 70)
        print("实验完成!")
        print("=" * 70)

        # 保存结果到文件
        import json

        # 保存简化结果
        summary = {
            'feature_info': results['feature_info'],
            'rf_avg_accuracy': results['rf_avg']['accuracy'],
            'rf_avg_precision': results['rf_avg']['precision'],
            'rf_avg_recall': results['rf_avg']['recall'],
            'rf_avg_macro_f1': results['rf_avg']['macro_f1'],
            'xgb_avg_accuracy': results['xgb_avg']['accuracy'],
            'xgb_avg_precision': results['xgb_avg']['precision'],
            'xgb_avg_recall': results['xgb_avg']['recall'],
            'xgb_avg_macro_f1': results['xgb_avg']['macro_f1'],
            'classes': results['classes'].tolist()
        }

        with open('experiment_results_smote_tomek_pca.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # 保存详细结果到CSV
        classes = results['classes']

        # RF详细结果
        rf_class_data = []
        for idx, cls in enumerate(classes):
            rf_class_data.append({
                'Class': cls,
                'Accuracy': results['rf_avg']['class_accuracy'][idx],
                'Precision': results['rf_avg']['class_precision'][idx],
                'Recall': results['rf_avg']['class_recall'][idx],
                'F1': results['rf_avg']['class_f1'][idx]
            })

        rf_class_df = pd.DataFrame(rf_class_data)
        rf_class_df.to_csv('rf_class_metrics_smote_tomek_pca.csv', index=False)

        # XGB详细结果
        xgb_class_data = []
        for idx, cls in enumerate(classes):
            xgb_class_data.append({
                'Class': cls,
                'Accuracy': results['xgb_avg']['class_accuracy'][idx],
                'Precision': results['xgb_avg']['class_precision'][idx],
                'Recall': results['xgb_avg']['class_recall'][idx],
                'F1': results['xgb_avg']['class_f1'][idx]
            })

        xgb_class_df = pd.DataFrame(xgb_class_data)
        xgb_class_df.to_csv('xgb_class_metrics_smote_tomek_pca.csv', index=False)

        print("\n结果已保存到以下文件:")
        print("1. 'experiment_results_smote_tomek_pca.json' - 实验摘要")
        print("2. 'rf_class_metrics_smote_tomek_pca.csv' - RF每类别详细指标")
        print("3. 'xgb_class_metrics_smote_tomek_pca.csv' - XGB每类别详细指标")

    except FileNotFoundError as e:
        print(f"文件加载错误: {e}")
        print("请检查文件路径是否正确:")
        print("1. D:/workspace/TTS-XGB/data/WSN-DS-main/train.parquet")
        print("2. D:/workspace/TTS-XGB/data/WSN-DS-main/test.parquet")
    except ModuleNotFoundError as e:
        print(f"缺少依赖库: {e}")
        print("请安装所需库: pip install pandas numpy scikit-learn xgboost imbalanced-learn")
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback

        traceback.print_exc()