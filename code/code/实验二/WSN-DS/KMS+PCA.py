import os
import glob
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans  # 新增KMeans导入
from sklearn.decomposition import PCA  # 新增PCA导入

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ====================== 基本配置 ======================
config = {
    # 新增KMS和PCA相关配置
    "n_clusters": 10,  # KMeans聚类数量
    "pca_components": 0.95,  # PCA保留的方差比例，也可以是整数（组件数量）
    "epochs": 100,
    "device": "cuda",
    "patience": 10,
    "num_classes": 5,
    "maj_threshold": 0.55,
    "test_size": 0.2,
    "random_state": 42,

    # RF参数
    "rf_n_estimators": 200,
    "rf_max_depth": 15,
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf": 2,
    "rf_class_weight": "balanced",

    # XGB参数
    "xgb_n_estimators": 200,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    "xgb_gamma": 0.1,
    "xgb_reg_alpha": 0.1,
    "xgb_reg_lambda": 1.0,
    "xgb_scale_pos_weight": 1,

    # Unknown检测阈值
    "unknown_threshold": 0.5,
    "energy_temperature": 1.0,
    "normal_fpr": 0.15,
    "allow_known_reject": 0.15,
}


# ====================== 固定随机种子 ======================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)


set_seed(config["random_state"])


# ====================== 读取 parquet 文件夹 ======================
def load_parquet_folder(folder_path: str) -> pd.DataFrame:
    parquet_files = [
        f for f in glob.glob(os.path.join(folder_path, "*"))
        if f.endswith(".parquet") and not f.endswith(".parquet.crc")
    ]
    print(f"📂 在 {folder_path} 中发现 {len(parquet_files)} 个有效 Parquet 文件")
    dfs = []
    for file in parquet_files:
        try:
            table = pq.read_table(file)
            df = table.to_pandas()
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ 跳过 {file}: {e}")
    if not dfs:
        raise RuntimeError(f"{folder_path} 下没有合法 parquet 数据")
    merged = pd.concat(dfs, ignore_index=True)
    return merged


# ====================== 数据预处理函数（新增KMS+PCA） ======================
def preprocess_with_kms_pca(X_train, X_val, X_test, config):
    """使用KMS和PCA对数据进行预处理"""
    # 1. KMeans聚类 - 为特征添加聚类信息
    print(f"🔍 执行KMeans聚类 (n_clusters={config['n_clusters']})...")
    kms = KMeans(n_clusters=config["n_clusters"], random_state=config["random_state"], n_init='auto')
    train_clusters = kms.fit_predict(X_train)
    val_clusters = kms.predict(X_val)
    test_clusters = kms.predict(X_test)

    # 将聚类结果作为新特征添加
    X_train_with_cluster = np.hstack([X_train, train_clusters.reshape(-1, 1)])
    X_val_with_cluster = np.hstack([X_val, val_clusters.reshape(-1, 1)])
    X_test_with_cluster = np.hstack([X_test, test_clusters.reshape(-1, 1)])

    # 2. PCA降维 - 减少特征维度
    print(f"🔍 执行PCA降维 (保留方差={config['pca_components']})...")
    pca = PCA(n_components=config["pca_components"], random_state=config["random_state"])
    X_train_pca = pca.fit_transform(X_train_with_cluster)
    X_val_pca = pca.transform(X_val_with_cluster)
    X_test_pca = pca.transform(X_test_with_cluster)

    print(f"📊 PCA降维完成: 原始维度={X_train_with_cluster.shape[1]}, "
          f"降维后维度={X_train_pca.shape[1]}")

    return X_train_pca, X_val_pca, X_test_pca, kms, pca


# ====================== 数据准备函数 ======================
def prepare_data(df: pd.DataFrame, label_col: str, feature_cols: list,
                 full2known: dict = None, is_test: bool = False):
    """准备数据，移除窗口机制，直接使用每个时间点的数据"""
    X = df[feature_cols].values.astype(np.float32)

    if is_test:
        # 测试集保留原始标签用于评估
        y_full = df[label_col].values.astype(int)
        if full2known:
            # 将已知类映射，Unknown保持原状
            y_known = np.array([full2known.get(label, -1) for label in y_full])
        else:
            y_known = y_full.copy()
        return X, y_known, y_full
    else:
        # 训练集使用映射后的标签
        y = df[label_col].values.astype(int)
        if full2known:
            y = np.array([full2known[label] for label in y])
        return X, y


# ====================== 能量分数计算 ======================
def energy_score(probs: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score: E(x) = -T * logsumexp(logits/T). Higher => more OOD/unknown."""
    logits = np.log(probs + 1e-12)
    x = logits / T
    m = np.max(x, axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.sum(np.exp(x - m), axis=1) + 1e-12)
    return -T * lse


# ====================== 模型训练函数 ======================
def train_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """训练并返回模型和结果"""
    model.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

    return {
        "model": model,
        "val_pred": y_val_pred,
        "val_proba": y_val_proba,
        "val_f1": val_f1,
        "val_true": y_val
    }


# ====================== 精简模型评估函数 ======================
def evaluate_model_simple(model_result, X_test, y_test_full, y_test_known,
                          full2known, known2full, UNKNOWN_FULL, model_name="Model"):
    """精简版模型评估，只输出关键指标"""
    model = model_result["model"]

    # 测试集预测
    test_proba = model.predict_proba(X_test)
    test_pred_known = model.predict(X_test)
    test_pred_full = np.array([known2full[int(k)] for k in test_pred_known], dtype=int)

    # 正常类ID
    normal_known_id = int(full2known.get(0, 0))

    # 计算各项分数
    pnormal_test = test_proba[:, normal_known_id]
    s_attack_test = 1.0 - pnormal_test
    E_test = energy_score(test_proba, T=config["energy_temperature"])

    # 在验证集上确定阈值
    val_proba = model_result["val_proba"]
    y_val_true = model_result["val_true"]
    E_val = energy_score(val_proba, T=config["energy_temperature"])
    pnormal_val = val_proba[:, normal_known_id]
    s_attack_val = 1.0 - pnormal_val

    # 确定阈值
    mask_norm_val = (y_val_true == normal_known_id)
    tau_energy = float(np.quantile(E_val, 1.0 - config["allow_known_reject"]))
    tau_attack = float(np.quantile(s_attack_val[mask_norm_val], 1.0 - config["normal_fpr"]))
    tau_p0_safe = float(np.quantile(pnormal_val[mask_norm_val], config["normal_fpr"]))

    # 应用Unknown检测
    final_pred = test_pred_full.copy()
    mask_high_energy = (E_test > tau_energy)
    mask_attack_like = (s_attack_test > tau_attack) & (pnormal_test < tau_p0_safe)
    mask_convert_to_unknown = (mask_high_energy | mask_attack_like) & (test_pred_full == 0)
    final_pred[mask_convert_to_unknown] = UNKNOWN_FULL

    # ------------ 计算四组关键指标 ------------
    results = {}

    # 1. 攻击检测指标
    gt_is_attack = (y_test_full != 0)
    pred_is_attack = (final_pred != 0)

    p_attack = precision_score(gt_is_attack.astype(int), pred_is_attack.astype(int), zero_division=0)
    r_attack = recall_score(gt_is_attack.astype(int), pred_is_attack.astype(int), zero_division=0)
    f1_attack = f1_score(gt_is_attack.astype(int), pred_is_attack.astype(int), zero_division=0)
    auc_attack = roc_auc_score(gt_is_attack.astype(int), s_attack_test)

    results["attack"] = {
        "precision": p_attack,
        "recall": r_attack,
        "f1": f1_attack,
        "auc": auc_attack
    }

    # 2. Unknown检测指标
    gt_is_unknown = (y_test_full == UNKNOWN_FULL)
    pred_is_unknown = (final_pred == UNKNOWN_FULL)

    if gt_is_unknown.sum() > 0:
        p_u = precision_score(gt_is_unknown.astype(int), pred_is_unknown.astype(int), zero_division=0)
        r_u = recall_score(gt_is_unknown.astype(int), pred_is_unknown.astype(int), zero_division=0)
        f1_u = f1_score(gt_is_unknown.astype(int), pred_is_unknown.astype(int), zero_division=0)
        auc_u = roc_auc_score(gt_is_unknown.astype(int), E_test)

        results["unknown"] = {
            "precision": p_u,
            "recall": r_u,
            "f1": f1_u,
            "auc": auc_u
        }
    else:
        results["unknown"] = None

    # 3. 已知类分类指标
    mask_known_eval = (~gt_is_unknown) & (~pred_is_unknown)

    if mask_known_eval.sum() > 0:
        yk = np.array([full2known.get(int(y), -1) for y in y_test_full[mask_known_eval]])
        pk = np.array([full2known.get(int(y), -1) for y in final_pred[mask_known_eval]])

        valid_mask = (yk != -1) & (pk != -1)
        yk_valid = yk[valid_mask]
        pk_valid = pk[valid_mask]

        if len(yk_valid) > 0:
            macro_f1_known = f1_score(yk_valid, pk_valid, average="macro", zero_division=0)
            coverage = float(mask_known_eval.mean())

            results["known"] = {
                "macro_f1": macro_f1_known,
                "coverage": coverage
            }
        else:
            results["known"] = None
    else:
        results["known"] = None

    # 4. Grayhole breakdown指标
    mask_gh = (y_test_full == UNKNOWN_FULL)
    n_gh = int(mask_gh.sum())

    if n_gh > 0:
        gh_as_unknown = int((mask_gh & pred_is_unknown).sum())
        gh_as_normal = int((mask_gh & (final_pred == 0)).sum())

        # 计算被预测为其他攻击的Grayhole
        other_attack_labels = [label for label in known2full.values()
                               if label != 0 and label != UNKNOWN_FULL]
        gh_as_other = 0
        other_attack_details = {}

        for attack_label in other_attack_labels:
            count = int((mask_gh & (final_pred == attack_label)).sum())
            gh_as_other += count
            if count > 0:
                other_attack_details[attack_label] = count

        results["grayhole"] = {
            "total": n_gh,
            "unknown": gh_as_unknown,
            "normal": gh_as_normal,
            "other_attack": gh_as_other,
            "other_details": other_attack_details
        }
    else:
        results["grayhole"] = None

    return results


# ====================== 主训练流程 ======================
def main():
    save_dir = Path("saved_models_tree_simple")
    save_dir.mkdir(exist_ok=True)
    start_time = datetime.now()
    run_tag = start_time.strftime("%Y%m%d-%H%M%S")

    # ------------ 1. 加载原始训练/测试数据 ------------
    print("🚀 开始加载 WSN-DS 数据集...")
    train_df = load_parquet_folder(r"D:\workspace\TTS-XGB\data\dataset_processing\newtrain.parquet")
    test_df = load_parquet_folder(r"D:\workspace\TTS-XGB\data\dataset_processing\test.parquet")

    ID_COL = "id"
    TIME_COL = "Time"
    LABEL_COL = "Attack_type"

    UNKNOWN_FULL = 2  # Grayhole 在你的数据里就是 2

    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TIME_COL, LABEL_COL]]

    # ------------ 2. 标准化 + 填充缺失 ------------
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()

    # ------------ 3. 已知类映射 ------------
    known_full = sorted([int(x) for x in train_df[LABEL_COL].unique().tolist()])
    assert UNKNOWN_FULL not in known_full, f"newtrain 里居然还有 Unknown={UNKNOWN_FULL}，先检查数据集！"

    full2known = {full: i for i, full in enumerate(known_full)}
    known2full = {i: full for full, i in full2known.items()}

    K = len(known_full)
    config["num_classes"] = K

    # ------------ 4. 准备数据 ------------
    # 训练数据
    X_train_all, y_train_all = prepare_data(train_df, LABEL_COL, feature_cols, full2known)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y_train_all
    )

    # 测试数据
    X_test, y_test_known, y_test_full = prepare_data(
        test_df, LABEL_COL, feature_cols, full2known, is_test=True
    )

    # 新增：应用KMS+PCA预处理
    X_train, X_val, X_test, kms_model, pca_model = preprocess_with_kms_pca(
        X_train, X_val, X_test, config
    )

    # ------------ 5. 训练所有模型 ------------
    print(f"\n{'=' * 60}")
    print("训练模型...")
    print(f"{'=' * 60}")

    all_results = {}

    # 1. 训练随机森林
    print("\n🌲 训练随机森林...")
    rf_model = RandomForestClassifier(
        n_estimators=config["rf_n_estimators"],
        max_depth=config["rf_max_depth"],
        min_samples_split=config["rf_min_samples_split"],
        min_samples_leaf=config["rf_min_samples_leaf"],
        class_weight=config["rf_class_weight"],
        random_state=config["random_state"],
        n_jobs=-1
    )

    rf_result = train_model(rf_model, X_train, y_train, X_val, y_val, "Random Forest")

    # 2. 训练XGBoost
    print("🌳 训练XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=config["xgb_n_estimators"],
        max_depth=config["xgb_max_depth"],
        learning_rate=config["xgb_learning_rate"],
        subsample=config["xgb_subsample"],
        colsample_bytree=config["xgb_colsample_bytree"],
        gamma=config["xgb_gamma"],
        reg_alpha=config["xgb_reg_alpha"],
        reg_lambda=config["xgb_reg_lambda"],
        scale_pos_weight=config["xgb_scale_pos_weight"],
        random_state=config["random_state"],
        use_label_encoder=False,
        eval_metric='mlogloss',
        device=config["device"],
        tree_method='gpu_hist' if config["device"] == "cuda" else 'auto'
    )

    xgb_result = train_model(xgb_model, X_train, y_train, X_val, y_val, "XGBoost")

    # ------------ 6. 在测试集上评估所有模型 ------------
    print(f"\n{'=' * 60}")
    print("评估结果")
    print(f"{'=' * 60}")

    # 评估随机森林
    rf_eval = evaluate_model_simple(
        rf_result, X_test, y_test_full, y_test_known,
        full2known, known2full, UNKNOWN_FULL, "Random Forest"
    )
    all_results["rf"] = rf_eval

    # 评估XGBoost
    xgb_eval = evaluate_model_simple(
        xgb_result, X_test, y_test_full, y_test_known,
        full2known, known2full, UNKNOWN_FULL, "XGBoost"
    )
    all_results["xgb"] = xgb_eval

    # ------------ 7. 输出四组关键指标 ------------
    print(f"\n{'=' * 60}")
    print("📊 四组关键指标对比")
    print(f"{'=' * 60}")

    # 1. 攻击检测指标
    print("\n🔍 1. 攻击检测性能:")
    print(f"{'模型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUROC':<8}")
    for model_name, results in all_results.items():
        if results["attack"]:
            attack = results["attack"]
            print(f"{model_name:<12} {attack['precision']:<8.4f} {attack['recall']:<8.4f} "
                  f"{attack['f1']:<8.4f} {attack['auc']:<8.4f}")

    # 2. Unknown检测指标
    print("\n🔍 2. Unknown检测性能:")
    print(f"{'模型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUROC':<8}")
    for model_name, results in all_results.items():
        if results["unknown"]:
            unknown = results["unknown"]
            print(f"{model_name:<12} {unknown['precision']:<8.4f} {unknown['recall']:<8.4f} "
                  f"{unknown['f1']:<8.4f} {unknown['auc']:<8.4f}")
        else:
            print(f"{model_name:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

    # 3. 已知类分类指标
    print("\n🔍 3. 已知类分类性能:")
    print(f"{'模型':<12} {'Macro-F1':<8} {'覆盖率':<8}")
    for model_name, results in all_results.items():
        if results["known"]:
            known = results["known"]
            print(f"{model_name:<12} {known['macro_f1']:<8.4f} {known['coverage']:<8.4f}")
        else:
            print(f"{model_name:<12} {'N/A':<8} {'N/A':<8}")

    # 4. Grayhole breakdown指标
    print("\n🔍 4. Grayhole去向分析:")
    print(f"{'模型':<12} {'总数':<8} {'Unknown':<8} {'Normal':<8} {'其他攻击':<8}")
    for model_name, results in all_results.items():
        if results["grayhole"]:
            grayhole = results["grayhole"]
            print(f"{model_name:<12} {grayhole['total']:<8} {grayhole['unknown']:<8} "
                  f"{grayhole['normal']:<8} {grayhole['other_attack']:<8}")

            # 如果有其他攻击详细分布，显示在下一行
            if grayhole["other_details"]:
                print(f"  详细分布: {grayhole['other_details']}")
        else:
            print(f"{model_name:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

    # ------------ 8. 保存结果 ------------
    print(f"\n💾 保存结果...")
    import joblib

    # 保存模型
    rf_model_path = save_dir / "rf_model.joblib"
    joblib.dump(rf_model, rf_model_path)

    xgb_model_path = save_dir / "xgb_model.joblib"
    joblib.dump(xgb_model, xgb_model_path)

    # 保存预处理相关模型（新增）
    kms_path = save_dir / "kms_model.joblib"
    joblib.dump(kms_model, kms_path)

    pca_path = save_dir / "pca_model.joblib"
    joblib.dump(pca_model, pca_path)

    # 保存scaler
    scaler_path = save_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # 保存完整结果
    results_path = save_dir / "results_summary.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"实验时间: {run_tag}\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"已知类数量: {K}\n")
        f.write(f"Unknown标签: {UNKNOWN_FULL}\n")
        f.write(f"KMS聚类数量: {config['n_clusters']}\n")  # 新增
        f.write(f"PCA降维后维度: {X_train.shape[1]}\n\n")  # 新增

        f.write("1. 攻击检测性能:\n")
        f.write(f"{'模型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUROC':<8}\n")
        for model_name, results in all_results.items():
            if results["attack"]:
                attack = results["attack"]
                f.write(f"{model_name:<12} {attack['precision']:<8.4f} {attack['recall']:<8.4f} "
                        f"{attack['f1']:<8.4f} {attack['auc']:<8.4f}\n")

        f.write("\n2. Unknown检测性能:\n")
        f.write(f"{'模型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUROC':<8}\n")
        for model_name, results in all_results.items():
            if results["unknown"]:
                unknown = results["unknown"]
                f.write(f"{model_name:<12} {unknown['precision']:<8.4f} {unknown['recall']:<8.4f} "
                        f"{unknown['f1']:<8.4f} {unknown['auc']:<8.4f}\n")
            else:
                f.write(f"{model_name:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}\n")

        f.write("\n3. 已知类分类性能:\n")
        f.write(f"{'模型':<12} {'Macro-F1':<8} {'覆盖率':<8}\n")
        for model_name, results in all_results.items():
            if results["known"]:
                known = results["known"]
                f.write(f"{model_name:<12} {known['macro_f1']:<8.4f} {known['coverage']:<8.4f}\n")
            else:
                f.write(f"{model_name:<12} {'N/A':<8} {'N/A':<8}\n")

        f.write("\n4. Grayhole去向分析:\n")
        f.write(f"{'模型':<12} {'总数':<8} {'Unknown':<8} {'Normal':<8} {'其他攻击':<8}\n")
        for model_name, results in all_results.items():
            if results["grayhole"]:
                grayhole = results["grayhole"]
                f.write(f"{model_name:<12} {grayhole['total']:<8} {grayhole['unknown']:<8} "
                        f"{grayhole['normal']:<8} {grayhole['other_attack']:<8}\n")

                if grayhole["other_details"]:
                    f.write(f"  详细分布: {grayhole['other_details']}\n")
            else:
                f.write(f"{model_name:<12} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}\n")

    print(f"✅ 结果已保存到: {results_path}")
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"⏱️  总耗时: {duration}")


if __name__ == "__main__":
    main()