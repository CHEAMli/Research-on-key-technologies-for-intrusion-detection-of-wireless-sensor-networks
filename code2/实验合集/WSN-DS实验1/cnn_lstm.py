"""
实验组合 2: Baseline + CNN-LSTM
预处理: 仅标准化
模型: CNN-LSTM
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from data_loader import WSNDataLoader
from preprocessing import BaselinePipeline
from models import get_model
from train_evaluate import train_and_evaluate


def run_experiment(n_repeats=5, save_results=True):
    """
    运行 Baseline + CNN-LSTM 实验
    
    Args:
        n_repeats: 重复次数
        save_results: 是否保存结果
    """
    print("="*80)
    print("实验: Baseline (仅标准化) + CNN-LSTM")
    print("="*80)
    
    results = []
    
    for repeat in range(n_repeats):
        seed = 42 + repeat
        print(f"\n{'='*80}")
        print(f"运行 {repeat+1}/{n_repeats} (随机种子: {seed})")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # 1. 加载数据
        print("\n[1/5] 加载数据...")
        loader = WSNDataLoader('./')
        X_train, y_train, X_val, y_val, X_test, y_test = loader.load_and_split()
        num_classes = loader.get_num_classes()
        
        # 2. 预处理 - Baseline (仅标准化)
        print("\n[2/5] 应用预处理: Baseline (仅标准化)")
        pipeline = BaselinePipeline()
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        y_train_processed = pipeline.y_resampled
        X_val_processed = pipeline.transform(X_val)
        X_test_processed = pipeline.transform(X_test)
        
        print(f"预处理后形状: {X_train_processed.shape}")
        
        # 3. 构建模型 - CNN-LSTM
        print("\n[3/5] 构建模型: CNN-LSTM")
        input_shape = (X_train_processed.shape[1],)
        model = get_model('cnn_lstm', input_shape, num_classes)
        
        # 4. 训练和评估
        print("\n[4/5] 训练模型...")
        test_metrics = train_and_evaluate(
            model=model,
            X_train=X_train_processed,
            y_train=y_train_processed,
            X_val=X_val_processed,
            y_val=y_val,
            X_test=X_test_processed,
            y_test=y_test,
            epochs=5,
            batch_size=128,
            patience=10,
            verbose=1
        )
        
        # 5. 记录结果
        result = {
            'experiment': 'Baseline_CNN_LSTM',
            'preprocessing': 'Baseline',
            'model': 'CNN_LSTM',
            'repeat': repeat + 1,
            'seed': seed,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_macro_f1': test_metrics['f1_score'],
            # 每个类别的指标
            'Normal_precision': test_metrics['Normal_precision'],
            'Normal_recall': test_metrics['Normal_recall'],
            'Normal_f1': test_metrics['Normal_f1'],
            'Blackhole_precision': test_metrics['Blackhole_precision'],
            'Blackhole_recall': test_metrics['Blackhole_recall'],
            'Blackhole_f1': test_metrics['Blackhole_f1'],
            'Flooding_precision': test_metrics['Flooding_precision'],
            'Flooding_recall': test_metrics['Flooding_recall'],
            'Flooding_f1': test_metrics['Flooding_f1'],
            'Grayhole_precision': test_metrics['Grayhole_precision'],
            'Grayhole_recall': test_metrics['Grayhole_recall'],
            'Grayhole_f1': test_metrics['Grayhole_f1'],
            'TDMA_precision': test_metrics['TDMA_precision'],
            'TDMA_recall': test_metrics['TDMA_recall'],
            'TDMA_f1': test_metrics['TDMA_f1']
        }
        results.append(result)
        
        print(f"\n[5/5] 结果: Acc={test_metrics['accuracy']:.4f}, "
              f"Prec={test_metrics['precision']:.4f}, "
              f"Rec={test_metrics['recall']:.4f}, "
              f"F1={test_metrics['f1_score']:.4f}")
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
    
    # 计算统计
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("实验汇总统计")
    print(f"{'='*80}")
    print(f"运行次数: {n_repeats}")
    print(f"Accuracy:  {df['test_accuracy'].mean():.4f} ± {df['test_accuracy'].std():.4f}")
    print(f"Precision: {df['test_precision'].mean():.4f} ± {df['test_precision'].std():.4f}")
    print(f"Recall:    {df['test_recall'].mean():.4f} ± {df['test_recall'].std():.4f}")
    print(f"Macro-F1:  {df['test_macro_f1'].mean():.4f} ± {df['test_macro_f1'].std():.4f}")
    
    # 打印每个类别的平均指标表格
    print(f"\n{'='*80}")
    print("各类别平均指标")
    print(f"{'='*80}")
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 80)
    classes = ['Normal', 'Blackhole', 'Flooding', 'Grayhole', 'TDMA']
    for cls in classes:
        prec = df[f'{cls}_precision'].mean()
        rec = df[f'{cls}_recall'].mean()
        f1 = df[f'{cls}_f1'].mean()
        print(f"{cls:<15} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    print("-" * 80)
    print(f"{'macro avg':<15} {df['test_precision'].mean():<12.4f} {df['test_recall'].mean():<12.4f} {df['test_macro_f1'].mean():<12.4f}")
    
    # 添加平均值行
    avg_result = {
        'experiment': 'Baseline_CNN_LSTM',
        'preprocessing': 'Baseline',
        'model': 'CNN_LSTM',
        'repeat': 'Average',
        'seed': 'N/A',
        'test_accuracy': df['test_accuracy'].mean(),
        'test_precision': df['test_precision'].mean(),
        'test_recall': df['test_recall'].mean(),
        'test_macro_f1': df['test_macro_f1'].mean(),
        'Normal_precision': df['Normal_precision'].mean(),
        'Normal_recall': df['Normal_recall'].mean(),
        'Normal_f1': df['Normal_f1'].mean(),
        'Blackhole_precision': df['Blackhole_precision'].mean(),
        'Blackhole_recall': df['Blackhole_recall'].mean(),
        'Blackhole_f1': df['Blackhole_f1'].mean(),
        'Flooding_precision': df['Flooding_precision'].mean(),
        'Flooding_recall': df['Flooding_recall'].mean(),
        'Flooding_f1': df['Flooding_f1'].mean(),
        'Grayhole_precision': df['Grayhole_precision'].mean(),
        'Grayhole_recall': df['Grayhole_recall'].mean(),
        'Grayhole_f1': df['Grayhole_f1'].mean(),
        'TDMA_precision': df['TDMA_precision'].mean(),
        'TDMA_recall': df['TDMA_recall'].mean(),
        'TDMA_f1': df['TDMA_f1'].mean()
    }
    df = pd.concat([df, pd.DataFrame([avg_result])], ignore_index=True)
    
    # 保存结果
    if save_results:
        os.makedirs('./results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'./results/baseline_cnn_lstm_{timestamp}.csv'
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {filepath}")
    
    return df


if __name__ == '__main__':
    run_experiment(n_repeats=1, save_results=True)
