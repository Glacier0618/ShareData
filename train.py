# -*- coding: utf-8 -*-
# @Time    : 2025/10/13 15:58
# @Author  : wp@1122
# @File    : train.py
# @Desc    :
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
from models.windnet import ThunderstormWindNet
from models.temporal_unet import ThunderstormWindNet3D
import torch.distributed as dist
import torch.nn.functional as F
import random
from datetime import datetime, timedelta
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ThunderWindDataset(Dataset):

    def __init__(self, rpath: str, ym_list: list, temporal_strategy="previous_time", use_ob=False, use_terrain=False, wind_mean=None, wind_std=None):
        self.samples = []
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.use_ob = use_ob
        self.use_terrain = use_terrain
        self.reflectivity_threshold = 35
        self.temporal_strategy = temporal_strategy  # "same_time", "previous", "any_previous"

        ymd_dir = os.listdir(rpath)
        ymd_dir.sort()
        for ymd in ymd_dir:
            if ymd[:len(ym_list[0])] not in ym_list:
                continue
            dpath = os.path.join(rpath, ymd)
            filelist = os.listdir(dpath)
            filelist.sort()
            self.samples.extend([os.path.join(dpath, f) for f in filelist if f.endswith(".npz")])
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        npz_file = self.samples[idx]
        # 2025060109_2025060106_0.npz
        patch_idx = os.path.basename(npz_file).split(".")[0].split("_")[-1]
        npz_data = np.load(npz_file)
        # [2, 12, 256, 256]
        input_data = npz_data["input"]
        # [1, 12, 256, 256]
        label_data = npz_data["label"]
        label_data = label_data[0]

        # [1, 3, 256, 256]
        ob_data = npz_data["ob"]
        ob_data = np.nan_to_num(ob_data, nan=0.0, posinf=0.0, neginf=0.0)
        ob_data = ob_data[0]

        # todo 需要增加地形数据
        # if self.use_terrain:
        #     ...
        # else:
        #     terrain = None

        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        label_data = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)

        input_data[input_data < 0] = 0
        label_data[label_data < 0] = 0
        ob_data[ob_data < 0] = 0

        x, g = input_data[0], input_data[1]  # x:反射率, g:阵风

        # 先归一化再掩码 避免阈值受归一化影响
        np.clip((x - 0) / 75., 0, 1, out=x)
        x[x < 0] = 0

        # 掩码策略
        if self.temporal_strategy == "same_time":
            # 相同时次的反射率掩码
            mask = (x >= (self.reflectivity_threshold / 75.0)).astype(np.float32)
        elif self.temporal_strategy == "previous_time":
            # 使用前一时次的反射率
            mask = np.ones_like(x)
            for t in range(12):
                if t == 0:
                    # 第一个时次使用自己（或者用实况的组合反射率？）
                    mask[t] = (x[t] >= (self.reflectivity_threshold / 75.0)).astype(np.float32)
                else:
                    # 使用时次t-1的反射率来预测时次t的阵风
                    mask[t] = (x[t - 1] >= (self.reflectivity_threshold / 75.0)).astype(np.float32)
        elif self.temporal_strategy == "any_previous":
            # 方案3：只要前面任意时次有强反射率就保留
            mask = np.zeros_like(x)
            cumulative_mask = np.zeros_like(x[0])
            for t in range(12):
                # 累积前面时次的强回波区域
                cumulative_mask = np.logical_or(cumulative_mask, x[t] >= (self.reflectivity_threshold / 75.0))
                mask[t] = cumulative_mask.astype(np.float32)

        # 应用掩码
        g = g * mask

        # 风速归一化
        if self.wind_mean is not None and self.wind_std is not None:
            wind_mask = g >= 0  # 只对有效风速数据归一化
            label_mask = label_data[0] >= 0

            g = np.where(wind_mask, (g - self.wind_mean) / self.wind_std, g)
            # 去除输出标签的归一化操作
            # label_data[0] = np.where(label_mask, (label_data[0] - self.wind_mean) / self.wind_std, label_data[0])

            if self.use_ob:
                ob_mask = ob_data[0] >= 0
                ob_data[0] = np.where(
                    ob_mask,
                    (ob_data[0] - self.wind_mean) / self.wind_std,
                    ob_data[0]
                )
                ob_data = np.nan_to_num(ob_data, nan=0.0, posinf=0.0, neginf=0.0)

        input_data = np.stack([x, g], axis=0)  # 重新组合
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        label_data = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(input_data), torch.from_numpy(label_data), torch.from_numpy(ob_data)


def calculate_wind_statistics_memory_efficient(dataset, wind_idx=1, use_ob=False):
    """
    内存友好的版本，适用于大数据集
    """
    print("开始计算风速统计量（内存友好版）...")

    sum_wind = 0.0
    sum_sq_wind = 0.0
    count = 0
    min_wind = float('inf')
    max_wind = float('-inf')
    total_samples = len(dataset)

    for i in tqdm(range(len(dataset))):
        try:
            if use_ob:
                input_data, label_data, ob_data = dataset[i]
            else:
                input_data, label_data = dataset[i]

            # 获取输入中的风速数据
            gust = input_data[wind_idx].numpy()
            label_wind = label_data[0].numpy()

            # 合并并过滤有效数据
            wind_data_to_process = []

            # 处理模式阵风数据
            gust_valid_mask = (gust != 0) & (~np.isnan(gust)) & (~np.isinf(gust))
            valid_gust = gust[gust_valid_mask]
            if len(valid_gust) > 0:
                wind_data_to_process.append(valid_gust)

            # 处理标签数据（雷暴大风实况）
            label_valid_mask = (label_wind != 0) & (~np.isnan(label_wind)) & (~np.isinf(label_wind))
            valid_label = label_wind[label_valid_mask]
            if len(valid_label) > 0:
                wind_data_to_process.append(valid_label)

            if use_ob:
                ob_wind = ob_data[0].numpy()
                ob_valid_mask = (ob_wind != 0) & (~np.isnan(ob_wind)) & (~np.isinf(ob_wind))
                valid_ob = ob_wind[ob_valid_mask]
                if len(valid_ob) > 0:
                    wind_data_to_process.append(valid_ob)

            if wind_data_to_process:
                all_valid_wind = np.concatenate([arr.flatten() for arr in wind_data_to_process])
                if len(all_valid_wind) > 0:
                    sum_wind += np.sum(all_valid_wind)
                    sum_sq_wind += np.sum(all_valid_wind ** 2)
                    count += len(all_valid_wind)
                    min_wind = min(min_wind, np.min(all_valid_wind))
                    max_wind = max(max_wind, np.max(all_valid_wind))

                    if i % 1000 == 0 and i > 0:
                        current_mean = sum_wind / count
                        current_std = np.sqrt(sum_sq_wind / count - current_mean ** 2)
                        print(f"已处理 {i}/{total_samples} 样本, 当前均值: {current_mean:.4f}, 当前标准差: {current_std:.4f}")

        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue

    if count == 0:
        raise ValueError("没有找到有效的风速数据")

    # 计算均值和标准差
    wind_mean = sum_wind / count
    wind_std = np.sqrt(sum_sq_wind / count - wind_mean ** 2)

    print(f"风速统计量计算完成:")
    print(f"总样本数: {total_samples}")
    print(f"有效数据点: {count}")
    print(f"风速均值 (mean): {wind_mean:.4f}")
    print(f"风速标准差 (std): {wind_std:.4f}")
    print(f"风速范围: [{min_wind:.4f}, {max_wind:.4f}]")

    return wind_mean, wind_std


def calculate_statistics_gpu(dataset, wind_idx=1, use_ob=False, batch_size=256, num_workers=8):
    """
    GPU加速的统计量计算
    """
    print("开始GPU加速计算统计量...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    total_sum = torch.tensor(0.0, device=device)
    total_sum_sq = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0, device=device)
    global_min = torch.tensor(float('inf'), device=device)
    global_max = torch.tensor(float('-inf'), device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if use_ob:
                input_data, label_data, ob_data = batch
            else:
                input_data, label_data = batch
                ob_data = None

            # 移动到GPU
            input_data = input_data.to(device, non_blocking=True)
            label_data = label_data.to(device, non_blocking=True)
            if ob_data is not None:
                ob_data = ob_data.to(device, non_blocking=True)

            # 处理当前批次
            batch_sum, batch_sum_sq, batch_count, batch_min, batch_max = process_batch_gpu(
                input_data, label_data, ob_data, wind_idx, use_ob
            )

            # 累加统计量
            total_sum += batch_sum
            total_sum_sq += batch_sum_sq
            total_count += batch_count
            global_min = torch.min(global_min, batch_min)
            global_max = torch.max(global_max, batch_max)

            if batch_idx % 100 == 0:
                current_mean = (total_sum / total_count).item()
                current_std = torch.sqrt(total_sum_sq / total_count - current_mean ** 2).item()
                print(f"已处理 {batch_idx * batch_size} 样本, 当前均值: {current_mean:.4f}, 当前标准差: {current_std:.4f}")

    # 移回CPU计算最终结果
    total_sum = total_sum.cpu().item()
    total_sum_sq = total_sum_sq.cpu().item()
    total_count = total_count.cpu().item()
    global_min = global_min.cpu().item()
    global_max = global_max.cpu().item()

    if total_count == 0:
        raise ValueError("没有找到有效数据")

    wind_mean = total_sum / total_count
    wind_std = np.sqrt(total_sum_sq / total_count - wind_mean ** 2)

    print(f"GPU计算完成:")
    print(f"总样本数: {len(dataset)}")
    print(f"有效数据点: {total_count}")
    print(f"风速均值: {wind_mean:.4f}")
    print(f"风速标准差: {wind_std:.4f}")
    print(f"风速范围: [{global_min:.4f}, {global_max:.4f}]")

    return wind_mean, wind_std


def process_batch_gpu(input_data, label_data, ob_data, wind_idx, use_ob):
    """ 在GPU上处理批次数据

    :param input_data: [C, T, H, W]
    :param label_data: [C, T, H, W]
    :param ob_data: [C, T, H, W]
    :param wind_idx:
    :param use_ob:
    :return:
    """
    batch_sum = torch.tensor(0.0, device=input_data.device)
    batch_sum_sq = torch.tensor(0.0, device=input_data.device)
    batch_count = torch.tensor(0, device=input_data.device)
    batch_min = torch.tensor(float('inf'), device=input_data.device)
    batch_max = torch.tensor(float('-inf'), device=input_data.device)

    # 处理阵风数据
    gust = input_data[wind_idx]
    gust_valid = (gust != 0) & (~torch.isnan(gust)) & (~torch.isinf(gust))
    valid_gust = gust[gust_valid]

    if len(valid_gust) > 0:
        batch_sum += torch.sum(valid_gust)
        batch_sum_sq += torch.sum(valid_gust ** 2)
        batch_count += len(valid_gust)
        batch_min = torch.min(batch_min, torch.min(valid_gust))
        batch_max = torch.max(batch_max, torch.max(valid_gust))

    # 处理标签数据
    label_wind = label_data[0]
    label_valid = (label_wind != 0) & (~torch.isnan(label_wind)) & (~torch.isinf(label_wind))
    valid_label = label_wind[label_valid]

    if len(valid_label) > 0:
        batch_sum += torch.sum(valid_label)
        batch_sum_sq += torch.sum(valid_label ** 2)
        batch_count += len(valid_label)
        batch_min = torch.min(batch_min, torch.min(valid_label))
        batch_max = torch.max(batch_max, torch.max(valid_label))

    # 处理观测数据
    if use_ob and ob_data is not None:
        ob_wind = ob_data[0]
        ob_valid = (ob_wind != 0) & (~torch.isnan(ob_wind)) & (~torch.isinf(ob_wind))
        valid_ob = ob_wind[ob_valid]

        if len(valid_ob) > 0:
            batch_sum += torch.sum(valid_ob)
            batch_sum_sq += torch.sum(valid_ob ** 2)
            batch_count += len(valid_ob)
            batch_min = torch.min(batch_min, torch.min(valid_ob))
            batch_max = torch.max(batch_max, torch.max(valid_ob))

    return batch_sum, batch_sum_sq, batch_count, batch_min, batch_max


class ThunderWindEvaluator:
    """
    雷暴大风模型评估器 - 专为二分类任务设计
    输入: [2, 256, 256] (组合反射率二值化 + 风速二值化)
    输出: [1, 256, 256] (风速≥13m/s的概率)
    """

    def __init__(self, model, wind_mean, wind_std, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.model.eval()

        # 阈值设置
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def load_model(self, model_path):
        """加载模型权重"""
        state_dict = torch.load(model_path, map_location=self.device)
        if hasattr(self.model, 'module'):  # DDP模型
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
        return True

    def predict_batch(self, data_loader, return_prob=True):
        """
        批量预测

        Args:
            data_loader: 数据加载器
            return_prob: 是否返回概率值

        Returns:
            predictions: 预测结果 [N, 1, 256, 256]
            targets: 真实标签 [N, 1, 256, 256]
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Predicting"):
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                outputs = self.model(future_forecast=inputs)

                if return_prob:
                    # 使用sigmoid将输出转换为概率
                    outputs = torch.sigmoid(outputs)

                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        return predictions, targets

    def calculate_binary_metrics(self, predictions, targets, threshold=0.5):
        """
        计算二分类指标

        Args:
            predictions: 预测概率 [N, 1, H, W]
            targets: 真实标签 [N, 1, H, W]
            threshold: 分类阈值

        Returns:
            metrics: 分类指标字典
        """
        # 二值化
        pred_binary = (predictions >= threshold).float()
        targets_binary = targets

        # 展平计算
        pred_flat = pred_binary.view(-1)
        target_flat = targets_binary.view(-1)

        # 计算混淆矩阵
        tp = torch.sum((pred_flat == 1) & (target_flat == 1)).item()
        fp = torch.sum((pred_flat == 1) & (target_flat == 0)).item()
        fn = torch.sum((pred_flat == 0) & (target_flat == 1)).item()
        tn = torch.sum((pred_flat == 0) & (target_flat == 0)).item()

        # 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 即POD
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0  # 空报率
        sr = fn / (tp + fn) if (tp + fn) > 0 else 0  # 漏报率
        ts = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # TS评分
        bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0  # 偏差
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            'threshold': threshold,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision,
            'Recall': recall,  # POD
            'F1_Score': f1,
            'FAR': far,  # 空报率
            'SR': sr,  # 漏报率
            'TS': ts,  # TS评分
            'BIAS': bias,  # 偏差
            'Accuracy': accuracy
        }

    def calculate_multiple_thresholds(self, predictions, targets):
        """
        计算多个阈值下的指标
        """
        results = {}
        for threshold in self.thresholds:
            results[f'threshold_{threshold}'] = self.calculate_binary_metrics(
                predictions, targets, threshold
            )
        return results

    def calculate_roc_auc(self, predictions, targets):
        """
        计算ROC曲线和AUC
        """
        pred_flat = predictions.view(-1).numpy()
        target_flat = targets.view(-1).numpy()

        # 过滤无效值
        valid_mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        if len(np.unique(target_flat)) < 2:
            return 0.5, None, None  # 如果只有一类，返回随机分类的AUC

        fpr, tpr, thresholds = roc_curve(target_flat, pred_flat)
        roc_auc = auc(fpr, tpr)

        return roc_auc, fpr, tpr

    def calculate_pr_auc(self, predictions, targets):
        """
        计算PR曲线和AUC
        """
        pred_flat = predictions.view(-1).numpy()
        target_flat = targets.view(-1).numpy()

        # 过滤无效值
        valid_mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        if len(np.unique(target_flat)) < 2:
            return 0.0, None, None

        precision, recall, thresholds = precision_recall_curve(target_flat, pred_flat)
        pr_auc = auc(recall, precision)

        return pr_auc, precision, recall

    def calculate_bce_loss(self, predictions, targets):
        """
        计算BCE损失
        """
        return F.binary_cross_entropy(predictions, targets).item()

    def calculate_ece(self, predictions, targets, n_bins=10):
        """
        计算预期校准误差 (ECE)
        """
        pred_flat = predictions.view(-1).numpy()
        target_flat = targets.view(-1).numpy()

        # 过滤无效值
        valid_mask = ~np.isnan(pred_flat) & ~np.isnan(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]

        # 分箱
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        calibration_details = []

        for i in range(n_bins):
            # 找到在bin内的样本
            in_bin = (pred_flat >= bin_lowers[i]) & (pred_flat < bin_uppers[i])
            bin_proportion = np.mean(in_bin)

            if bin_proportion > 0:
                # 计算这个bin的准确率
                bin_accuracy = np.mean(target_flat[in_bin])
                bin_confidence = np.mean(pred_flat[in_bin])

                ece += bin_proportion * np.abs(bin_accuracy - bin_confidence)

                calibration_details.append({
                    'bin': i,
                    'accuracy': bin_accuracy,
                    'confidence': bin_confidence,
                    'proportion': bin_proportion
                })

        return ece, calibration_details

    def comprehensive_evaluation(self, data_loader, save_path=None):
        """
        综合评估模型性能
        """
        print("Starting comprehensive evaluation...")

        # 1. 获取预测结果
        predictions, targets = self.predict_batch(data_loader, return_prob=True)
        self._save_result_to_nc(predictions, targets)
        # 保存为nc

        # 2. 计算多个阈值下的指标
        # threshold_metrics = self.calculate_multiple_thresholds(predictions, targets)

        # 3. 计算ROC和PR曲线
        # roc_auc, fpr, tpr = self.calculate_roc_auc(predictions, targets)
        # pr_auc, precision, recall = self.calculate_pr_auc(predictions, targets)

        # 4. 计算BCE损失
        # bce_loss = self.calculate_bce_loss(predictions, targets)

        # 5. 计算校准误差
        # ece, calibration_details = self.calculate_ece(predictions, targets)

        # 6. 整合结果
        # results = {
        #     'predictions': predictions,
        #     'targets': targets,
        #     'threshold_metrics': threshold_metrics,
        #     # 'roc_auc': roc_auc,
        #     # 'pr_auc': pr_auc,
        #     # 'bce_loss': bce_loss,
        #     'ece': ece,
        #     'calibration_details': calibration_details
        # }

        # 7. 打印结果
        # self._print_evaluation_results(results)

        # 8. 保存结果和图表
        if save_path:
            self._save_evaluation_results(results, save_path)
            self._plot_evaluation_results(results, save_path)

        # return results


    def _save_result_to_nc(self, predictions, targets):
        np.savez_compressed("./result", pred=predictions, label=targets)
        # reso = 0.03
        # # 插值区域
        # lon1d = np.arange(110, 120 + reso / 2, reso)
        # lat1d = np.arange(42, 31 - reso / 2, -reso)
        # for idx, pred in enumerate(predictions):
        #     target = targets[idx]



    def _print_evaluation_results(self, results):
        """打印评估结果"""
        print("\n" + "=" * 80)
        print("                  THUNDERSTORM WIND BINARY CLASSIFICATION EVALUATION")
        print("=" * 80)

        print(f"\n--- OVERALL METRICS ---")
        print(f"BCE Loss: {results['bce_loss']:.4f}")
        print(f"ROC AUC:  {results['roc_auc']:.4f}")
        print(f"PR AUC:   {results['pr_auc']:.4f}")
        print(f"ECE:      {results['ece']:.4f}")

        print(f"\n--- THRESHOLD METRICS (TS: Threat Score, POD: Probability of Detection) ---")
        print("Thresh    TS      POD     Precision FAR     SR      BIAS    F1      Accuracy")
        print("-" * 85)

        for thresh in self.thresholds:
            metrics = results['threshold_metrics'][f'threshold_{thresh}']
            print(f"{thresh:6.1f}   {metrics['TS']:7.3f} {metrics['Recall']:7.3f} "
                  f"{metrics['Precision']:7.3f} {metrics['FAR']:7.3f} "
                  f"{metrics['SR']:7.3f} {metrics['BIAS']:7.3f} "
                  f"{metrics['F1_Score']:7.3f} {metrics['Accuracy']:7.3f}")

    def _save_evaluation_results(self, results, save_path):
        """保存评估结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存阈值指标为CSV
        threshold_data = []
        for thresh in self.thresholds:
            metrics = results['threshold_metrics'][f'threshold_{thresh}']
            threshold_data.append(metrics)

        df_threshold = pd.DataFrame(threshold_data)
        df_threshold.to_csv(f"{save_path}_threshold_metrics.csv", index=False)

        # 保存总体指标
        overall_metrics = {
            'BCE_Loss': results['bce_loss'],
            'ROC_AUC': results['roc_auc'],
            'PR_AUC': results['pr_auc'],
            'ECE': results['ece']
        }
        df_overall = pd.DataFrame([overall_metrics])
        df_overall.to_csv(f"{save_path}_overall_metrics.csv", index=False)

        print(f"\nResults saved to {save_path}*")

    def _plot_evaluation_results(self, results, save_path):
        """绘制评估图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. ROC曲线
        roc_auc, fpr, tpr = results['roc_auc'], results.get('fpr'), results.get('tpr')
        if fpr is not None and tpr is not None:
            axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2,
                            label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 0].set_xlim([0.0, 1.0])
            axes[0, 0].set_ylim([0.0, 1.05])
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend(loc="lower right")

        # 2. PR曲线
        pr_auc, precision, recall = results['pr_auc'], results.get('precision'), results.get('recall')
        if precision is not None and recall is not None:
            axes[0, 1].plot(recall, precision, color='blue', lw=2,
                            label=f'PR curve (AUC = {pr_auc:.3f})')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend(loc="upper right")

        # 3. TS评分随阈值变化
        thresholds = self.thresholds
        ts_scores = [results['threshold_metrics'][f'threshold_{t}']['TS'] for t in thresholds]
        axes[0, 2].plot(thresholds, ts_scores, 'bo-')
        axes[0, 2].set_xlabel('Threshold')
        axes[0, 2].set_ylabel('TS Score')
        axes[0, 2].set_title('TS Score vs Threshold')
        axes[0, 2].grid(True)

        # 4. 校准曲线
        ece_details = results['calibration_details']
        if ece_details:
            accuracies = [detail['accuracy'] for detail in ece_details]
            confidences = [detail['confidence'] for detail in ece_details]

            axes[1, 0].plot(confidences, accuracies, 'bo-', label='Model')
            axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title(f'Calibration Curve (ECE = {results["ece"]:.3f})')
            axes[1, 0].legend()

        # 5. 概率分布
        predictions_flat = results['predictions'].view(-1).numpy()
        targets_flat = results['targets'].view(-1).numpy()

        axes[1, 1].hist(predictions_flat[targets_flat == 0], bins=50, alpha=0.7,
                        label='Negative', color='blue', density=True)
        axes[1, 1].hist(predictions_flat[targets_flat == 1], bins=50, alpha=0.7,
                        label='Positive', color='red', density=True)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Probability Distribution')
        axes[1, 1].legend()

        # 6. 最佳阈值混淆矩阵
        best_threshold = 0.5  # 默认使用0.5
        best_metrics = results['threshold_metrics'][f'threshold_{best_threshold}']
        cm = [[best_metrics['TN'], best_metrics['FP']],
              [best_metrics['FN'], best_metrics['TP']]]

        im = axes[1, 2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 2].set_title(f'Confusion Matrix (Threshold={best_threshold})')
        plt.colorbar(im, ax=axes[1, 2])

        # 添加数值标签
        thresh = np.array(cm).max() / 2.
        for i in range(2):
            for j in range(2):
                axes[1, 2].text(j, i, format(cm[i][j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i][j] > thresh else "black")

        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['Pred 0', 'Pred 1'])
        axes[1, 2].set_yticklabels(['True 0', 'True 1'])

        plt.tight_layout()
        plt.savefig(f"{save_path}_evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()

    def predict_single(self, npz_file):
        """
        单样本预测 - 用于业务部署
        """
        # patch_idx = os.path.basename(npz_file).split(".")[0].split("_")[-1]
        npz_data = np.load(npz_file)
        # [2, 12, 256, 256]
        input_data = npz_data["input"]
        # [1, 12, 256, 256]
        label_data = npz_data["label"]
        label_data = label_data[0]

        # [1, 3, 256, 256]
        ob_data = npz_data["ob"]
        ob_data = np.nan_to_num(ob_data, nan=0.0, posinf=0.0, neginf=0.0)
        ob_data = ob_data[0]

        # todo 需要增加地形数据
        # if self.use_terrain:
        #     ...
        # else:
        #     terrain = None

        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        label_data = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)

        x, g = input_data[0], input_data[1]  # x:反射率, g:阵风

        # 先归一化再掩码 避免阈值受归一化影响
        np.clip((x - 0) / 75., 0, 1, out=x)
        x[x < 0] = 0

        # 掩码策略
        # 使用前一时次的反射率
        mask = np.ones_like(x)
        for t in range(12):
            if t == 0:
                # 第一个时次使用自己（或者用实况的组合反射率？）
                mask[t] = (x[t] >= (35 / 75.0)).astype(np.float32)
            else:
                # 使用时次t-1的反射率来预测时次t的阵风
                mask[t] = (x[t - 1] >= (35 / 75.0)).astype(np.float32)

        # 应用掩码
        g = g * mask

        # 风速归一化
        if self.wind_mean is not None and self.wind_std is not None:
            wind_mask = g >= 0  # 只对有效风速数据归一化
            label_mask = label_data[0] >= 0

            g = np.where(wind_mask, (g - self.wind_mean) / self.wind_std, g)
            label_data[0] = np.where(label_mask, (label_data[0] - self.wind_mean) / self.wind_std, label_data[0])


            ob_mask = ob_data[0] >= 0
            ob_data[0] = np.where(
                ob_mask,
                (ob_data[0] - self.wind_mean) / self.wind_std,
                ob_data[0]
            )
            ob_data = np.nan_to_num(ob_data, nan=0.0, posinf=0.0, neginf=0.0)

        input_data = np.stack([x, g], axis=0)  # 重新组合
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        label_data = np.nan_to_num(label_data, nan=0.0, posinf=0.0, neginf=0.0)

        input_data = torch.from_numpy(input_data).float()
        # label_data = torch.from_numpy(label_data).float()
        ob_data = torch.from_numpy(ob_data).float()
        self.model.eval()

        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).float()

        with torch.no_grad():
            input_tensor = input_data.unsqueeze(0).to(self.device)
            # label_tensor = label_data.unsqueeze(0).to(self.device)
            ob_tensor = ob_data.unsqueeze(0).to(self.device)

            output = self.model(ob_tensor, input_tensor)

            output = output[0].squeeze(0).cpu()

        filename = os.path.basename(npz_file)
        sv_dir = os.path.join(os.path.dirname(__file__), "predict")
        os.makedirs(sv_dir, exist_ok=True)
        sv_file = os.path.join(sv_dir, filename)
        np.savez_compressed(sv_file, pred=output, label=label_data)
        print(f"success:{sv_file}")


    def create_test_dataloader(self, test_dataset, batch_size=16):
        """创建测试数据加载器"""
        return DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )


def generate_date_splits(date_ranges=None,
                         train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                         random_seed=42):
    """
    生成随机的日期划分
    Args:
        date_ranges: 日期范围列表，例如 [("20240601", "20240831"), ("20250601", "20250831")]
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    # 设置随机种子
    random.seed(random_seed)

    # 生成所有日期
    all_dates = []

    if date_ranges is None:
        # 默认使用连续日期（向后兼容）
        date_ranges = [("20240501", "20240831")]

    for start_date, end_date in date_ranges:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        current = start
        while current <= end:
            all_dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

    print(f"总日期数: {len(all_dates)}天")
    print(f"日期范围: {date_ranges}")

    # 显示数据分布
    year_month_counts = {}
    for date_str in all_dates:
        ym = date_str[:6]  # YYYYMM
        year_month_counts[ym] = year_month_counts.get(ym, 0) + 1

    print("\n各月份数据分布:")
    for ym in sorted(year_month_counts.keys()):
        print(f"  {ym}: {year_month_counts[ym]}天")

    # 随机打乱
    random.shuffle(all_dates)

    # 计算各集合大小
    total_days = len(all_dates)
    train_size = int(total_days * train_ratio)
    val_size = int(total_days * val_ratio)

    # 划分
    train_dates = all_dates[:train_size]
    val_dates = all_dates[train_size:train_size + val_size]
    test_dates = all_dates[train_size + val_size:]

    # 转换为月份格式（YM）
    def dates_to_ym_list(dates):
        ym_set = set()
        for date_str in dates:
            ym_set.add(date_str[:6])  # 取前6位 "YYYYMM"
        return sorted(list(ym_set))

    train_ym = dates_to_ym_list(train_dates)
    val_ym = dates_to_ym_list(val_dates)
    test_ym = dates_to_ym_list(test_dates)

    # 打印统计信息
    print(f"\n划分结果:")
    print(f"训练集: {len(train_dates)}天, 月份: {train_ym}")
    print(f"验证集: {len(val_dates)}天, 月份: {val_ym}")
    print(f"测试集: {len(test_dates)}天, 月份: {test_ym}")

    # 检查各月份在三个集合中的分布
    print(f"\n各月份在训练/验证/测试集中的分布:")
    all_ym = sorted(set(train_ym + val_ym + test_ym))
    for ym in all_ym:
        ym_train = sum(1 for d in train_dates if d.startswith(ym))
        ym_val = sum(1 for d in val_dates if d.startswith(ym))
        ym_test = sum(1 for d in test_dates if d.startswith(ym))
        ym_total = ym_train + ym_val + ym_test
        print(f"  {ym}: 训练{ym_train}天({ym_train / ym_total * 100:.1f}%), "
              f"验证{ym_val}天({ym_val / ym_total * 100:.1f}%), "
              f"测试{ym_test}天({ym_test / ym_total * 100:.1f}%)")

    return train_ym, val_ym, test_ym, train_dates, val_dates, test_dates


class Train:
    def __init__(self, model, criterion, train_dataset, val_dataset, lr=1e-3, batch_size=32,
                 epochs=100, patience=20, ckpt_name="UNet_ConvLSTM_256x256", use_ddp=False, checkpoint_dir=None):
        self.model = model.float()
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.ckpt_name = ckpt_name
        self.patience = patience
        self.use_ddp = use_ddp

        # 设置检查点目录
        if checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 训练记录
        self.train_losses = []
        self.validation_losses = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.stop_training = False

        # 设备
        self.device = self._setup_device()
        self.model.to(self.device)
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # 学习率衰减因子
            patience=5,  # 等待5个epoch没有改善后再衰减
            verbose=True,  # 打印学习率变化
            min_lr=1e-7  # 最小学习率
        )
        # DDP 设置
        if use_ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=False
            )
        # 数据加载器
        self.train_loader, self.val_loader = self._setup_dataloaders()

        # 载入模型
        self.best_model_path = os.path.join(self.checkpoint_dir, f'{self.ckpt_name}.pth')

    def load_best_model(self):
        """加载最佳模型"""
        if os.path.exists(self.best_model_path):
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            if self.use_ddp:
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self._log(f"Loaded best model from {self.best_model_path}")
            return True
        else:
            self._log(f"Best model not found at {self.best_model_path}")
            return False

    def _setup_device(self):
        if self.use_ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            return torch.device("cuda", local_rank)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_dataloaders(self):
        if self.use_ddp:
            train_sampler = DistributedSampler(self.train_dataset)
            train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                num_workers=2, pin_memory=False
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=self.batch_size * 2, shuffle=False,
                num_workers=1, pin_memory=False
            )
        else:
            train_loader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=self.batch_size * 2, shuffle=False,
                num_workers=2, pin_memory=True
            )
        return train_loader, val_loader

    def _log(self, *args):
        if not self.use_ddp or dist.get_rank() == 0:
            print(*args)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_records = {'loss': []}
        rank = dist.get_rank() if self.use_ddp else 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}", disable=(rank != 0))):
            inputs, targets, obs = batch
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).float()
            obs = obs.to(self.device).float()

            # print("输入数据大小", inputs.shape, targets.shape, obs.shape)
            outputs = self.model(obs, inputs)
            losses = self.criterion(outputs, targets)

            # y_reg, y_cls = self.model(obs, inputs)  # y_reg:[B,12,H,W] (m/s)
            # losses, parts = self.criterion(y_reg, y_cls, targets, mask=None)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            epoch_records['loss'].append(losses.item())

        return epoch_records

    def valid_epoch(self, epoch):
        self.model.eval()
        epoch_records = {'loss': []}
        rank = dist.get_rank() if self.use_ddp else 0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"[Val] Epoch {epoch}", disable=(rank != 0))):
            with torch.no_grad():
                inputs, targets, obs = batch
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()
                obs = obs.to(self.device).float()
                outputs = self.model(obs, inputs)
                losses = self.criterion(outputs, targets)

                # y_reg, y_cls = self.model(obs, inputs)  # y_reg:[B,12,H,W] (m/s)
                # losses, parts = self.criterion(y_reg, y_cls, targets, mask=None)

                epoch_records['loss'].append(losses.item())

        return epoch_records


    def run(self):
        if not self.use_ddp or dist.get_rank() == 0:
            self._log("模型第一个参数所在设备:", next(self.model.parameters()).device)
            # 数据类型检查
            sample_input = next(iter(self.train_loader))[0]
            self._log(f"输入数据数据类型: {sample_input.dtype}")
            self._log(f"模型参数数据类型: {next(self.model.parameters()).dtype}")

        for epoch in range(self.epochs):
            if self.use_ddp:
                dist.barrier()  # 确保所有进程同步
                if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

            if self.stop_training:
                self._log("Stopping training at epoch:", epoch)
                break

            # 训练和验证
            train_records = self.train_epoch(epoch)
            valid_records = self.valid_epoch(epoch)

            # 只在 rank0 处理结果和保存模型
            if not self.use_ddp or dist.get_rank() == 0:
                train_loss = np.mean(train_records['loss'])
                val_loss = np.mean(valid_records['loss'])

                self.train_losses.append(train_loss)
                self.validation_losses.append(val_loss)

                improved = val_loss < self.best_val_loss - 1e-6
                if improved:
                    self.best_val_loss = val_loss
            else:
                improved = False

            # 广播早停信号到所有卡
            if self.use_ddp:
                improved_tensor = torch.tensor(bool(improved), dtype=torch.uint8, device=self.device)
                dist.broadcast(improved_tensor, src=0)
                improved = bool(improved_tensor.item())

            # 应用学习率调度器
            if not self.use_ddp or dist.get_rank() == 0:
                self.scheduler.step(val_loss)  # 根据验证损失调整学习率

            if improved:
                self.epochs_no_improve = 0
                # 仅 rank0 保存模型
                if not self.use_ddp or dist.get_rank() == 0:
                    model_to_save = self.model.module if self.use_ddp else self.model
                    torch.save(
                        model_to_save.state_dict(),
                        os.path.join(self.checkpoint_dir, f'{self.ckpt_name}.pth')
                    )
                    self._log(f"Saved Best Model with Val Loss={val_loss:.4f}")
            else:
                self.epochs_no_improve += 1
                if not self.use_ddp or dist.get_rank() == 0:
                    self._log(f"No improvement for {self.epochs_no_improve} epochs.")

            if not self.use_ddp or dist.get_rank() == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

            if self.patience == self.epochs_no_improve:
                self.stop_training = True

        # 训练完成
        self._log("Training losses:", self.train_losses)
        self._log("Validation losses:", self.validation_losses)
        self._log("Training completed.")

        if self.use_ddp:
            dist.destroy_process_group()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.validation_losses,
            'best_val_loss': self.best_val_loss
        }


if __name__ == '__main__':
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        # torchrun --nproc_per_node=8 train.py
        # python -m torch.distributed.run --nproc_per_node=8 train.py
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
    # 数据根目录
    data_root = r"/public/home/huashan/data/thunderwind212/npz_chn_256"
    # 数据集日期划分 固定生成
    """
        分布不一致: 训练集和验证集的风速统计量差异明显
        均值差异: 训练集均值(3.32) vs 验证集均值(2.47)
        标准差差异: 训练集std(4.14) vs 验证集std(3.19)
    """
    date_ranges = [
        # ("20240601", "20240831"),  # 2024年5-8月
        # ("20250501", "20250831")   # 2025年5-8月 Mean: 1.0862  Std: 3.0150
        ("20250501", "20250531")   # 小数据集 Mean: 1.0441, Std: 2.9616
    ]
    train_ym, val_ym, test_ym, train_dates, val_dates, test_dates = generate_date_splits(
        date_ranges=date_ranges,
        train_ratio=0.5,
        val_ratio=0.3,
        test_ratio=0.2,
        random_seed=42
    )
    # # 1、计算输入和标签样本的mean std  对meso的阵风 + n 然后再过滤
    # 训练集
    # train_dataset = ThunderWindDataset(rpath=data_root, ym_list=train_dates, use_ob=True)
    # # train_mean, train_std = calculate_wind_statistics_memory_efficient(train_dataset, use_ob=True)
    # train_mean, train_std = calculate_statistics_gpu(train_dataset, use_ob=True, num_workers=16)
    # print(f"训练集 - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
    #
    # # # 验证集
    # val_dataset = ThunderWindDataset(rpath=data_root, ym_list=val_dates, use_ob=True)
    # # # 计算验证集的风速统计量（用于对比）
    # # val_mean, val_std = calculate_wind_statistics_memory_efficient(val_dataset, use_ob=True)
    # val_mean, val_std = calculate_statistics_gpu(val_dataset, use_ob=True, num_workers=16)
    # print(f"验证集 - Mean: {val_mean:.4f}, Std: {val_std:.4f}")
    # quit()

    # 2、模型训练
    # 20250501-20250831的均值
    train_mean = 1.0862
    train_std = 3.0150
    config = {
        'use_past_wind': True,  # 是否使用过去3小时实况
        'use_terrain': False,  # 是否使用地形数据
    }
    model = ThunderstormWindNet()
    # model = ThunderstormWindNet3D(base_channels=32, thresholds=(15, 20, 25, 30), use_topo=False)
    train_dataset = ThunderWindDataset(data_root, use_ob=True, ym_list=train_dates, wind_mean=train_mean, wind_std=train_std)
    val_dataset = ThunderWindDataset(data_root, use_ob=True, ym_list=val_dates, wind_mean=train_mean, wind_std=train_std)
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    from losses import CompositeLoss
    criterion = CompositeLoss(
        huber_delta=1.0,
        focal_gamma=2.0,
        focal_alpha=0.75,
        thresholds=(15, 20, 25, 30),
        lambda_cls=0.3,  # 若只做回归可设 0.0
    )
    trainer = Train(model, criterion, train_dataset, val_dataset, lr=1e-4, batch_size=8, epochs=300, use_ddp=use_ddp)
    results = trainer.run()
    quit()

    # # 3、评估
    # test_dataset = ThunderWindDataset(data_root, ym_list=test_dates, wind_mean=train_mean, wind_std=train_std)
    # 创建评估器
    evaluator = ThunderWindEvaluator(model, train_mean, train_std, device='cpu')
    evaluator.load_model("checkpoints/UNet_ConvLSTM_256x256.pth")
    # 预测单个文件
    evaluator.predict_single("2025050100_2025043018_114.npz")
    # # 创建测试数据加载器
    # test_loader = evaluator.create_test_dataloader(test_dataset, batch_size=16)
    # # 综合评估
    # results = evaluator.comprehensive_evaluation(
    #     test_loader,
    #     save_path=None
    # )
    # print(results)