"""
实验: Baseline预处理 + CNN-LSTM模型 (IoT数据集 - 十分类)
数据集: Train_Test_IoT_datasets (使用type列，10个攻击类别)
类别: normal, backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss
"""
import numpy as np
import pandas as pd
from datetime import datetime
from data_loader_iot import IoTDataLoader
from preprocessing import BaselinePipeline
from models import build_cnn_lstm_model
from train_evaluate import train_and_evaluate

# 实验配置
EXPERIMENT_NAME = "IoT_Baseline_CNN_LSTM"
NUM_RUNS = 5
EPOCHS = 1
BATCH_SIZE = 128
RESULTS_DIR = "./results2"

print(f"{'='*60}")
print(f"实验: {EXPERIMENT_NAME}")
print(f"数据集: Train_Test_IoT_datasets (十分类)")
print(f"类别: normal, backdoor, ddos, dos, injection,")
print(f"      mitm, password, ransomware, scanning, xss")
print(f"{'='*60}\n")

# 加载数据
loader = IoTDataLoader()
X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_split()

# 获取类别名称（按编码顺序）
target_names = list(loader.label_encoder.classes_)
print(f"\n类别编码顺序: {target_names}")

# 存储结果
all_results = []

# 运行多次实验
for run_idx in range(NUM_RUNS):
    seed = 42 + run_idx
    print(f"\n{'='*60}")
    print(f"运行 {run_idx + 1}/{NUM_RUNS} (随机种子: {seed})")
    print(f"{'='*60}")
    
    # 设置随机种子
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    
    # 数据预处理
    print("\n[1/3] 数据预处理...")
    pipeline = BaselinePipeline()
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)
    y_train_processed = pipeline.y_resampled  # 使用预处理后的标签（对于Baseline与原标签相同）
    
    # 构建模型
    print("\n[2/3] 构建CNN-LSTM模型...")
    num_classes = loader.get_num_classes()
    model = build_cnn_lstm_model(
        input_shape=(X_train_processed.shape[1],),
        num_classes=num_classes
    )
    
    # 训练和评估
    print("\n[3/3] 训练和评估...")
    results = train_and_evaluate(
        model=model,
        X_train=X_train_processed,
        y_train=y_train_processed,
        X_val=X_val_processed,
        y_val=y_val,
        X_test=X_test_processed,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        target_names=target_names  # 传入IoT数据集的类别名称
    )
    
    # 添加运行信息
    results['run'] = run_idx + 1
    results['seed'] = seed
    all_results.append(results)
    
    print(f"\n运行 {run_idx + 1} 结果:")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1_score']:.4f}")

# 汇总结果
print(f"\n{'='*60}")
print("所有运行完成! 汇总结果:")
print(f"{'='*60}")

results_df = pd.DataFrame(all_results)
print(f"\n平均性能 (±标准差):")
print(f"  准确率: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  精确率: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
print(f"  召回率: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
print(f"  F1分数: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")

# 保存结果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"{RESULTS_DIR}/iot_baseline_cnn_lstm_{timestamp}.csv"
results_df.to_csv(output_file, index=False)
print(f"\n结果已保存到: {output_file}")
