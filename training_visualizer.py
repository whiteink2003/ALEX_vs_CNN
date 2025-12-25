import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import random
import shutil
from tqdm import tqdm
from PIL import Image
import warnings

from data_preprocessor import DataPreprocessor

warnings.filterwarnings('ignore')


class TrainingVisualizer:
  def __init__(self, history=None, model_name="Model"):
    """
    初始化训练可视化器

    参数：
    - history: 训练历史对象
    - model_name: 模型名称
    """
    self.history = history
    self.model_name = model_name
    self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

  def set_history(self, history):
    """
    设置训练历史

    参数：
    - history: 训练历史
    """
    self.history = history

  def plot_training_history(self, save_path=None):
    """
    绘制训练历史

    参数：
    - save_path: 保存路径
    """
    if self.history is None:
      raise ValueError("训练历史为空，请先设置history")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 获取历史记录中的指标
    history_dict = self.history.history

    # 1. 损失曲线
    axes[0].plot(history_dict['loss'], label='Training Loss',
                 color=self.colors[0], linewidth=2)
    axes[0].plot(history_dict['val_loss'], label='Validation Loss',
                 color=self.colors[1], linewidth=2)
    axes[0].set_title(f'{self.model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 准确率曲线
    axes[1].plot(history_dict['accuracy'], label='Training Accuracy',
                 color=self.colors[0], linewidth=2)
    axes[1].plot(history_dict['val_accuracy'], label='Validation Accuracy',
                 color=self.colors[1], linewidth=2)
    axes[1].set_title(f'{self.model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 精确率曲线（如果存在）
    if 'precision' in history_dict:
      axes[2].plot(history_dict['precision'], label='Training Precision',
                   color=self.colors[0], linewidth=2)
      if 'val_precision' in history_dict:
        axes[2].plot(history_dict['val_precision'], label='Validation Precision',
                     color=self.colors[1], linewidth=2)
      axes[2].set_title(f'{self.model_name} - Precision Curves', fontsize=14, fontweight='bold')
      axes[2].set_xlabel('Epochs')
      axes[2].set_ylabel('Precision')
      axes[2].legend()
      axes[2].grid(True, alpha=0.3)

    # 4. 召回率曲线（如果存在）
    if 'recall' in history_dict:
      axes[3].plot(history_dict['recall'], label='Training Recall',
                   color=self.colors[0], linewidth=2)
      if 'val_recall' in history_dict:
        axes[3].plot(history_dict['val_recall'], label='Validation Recall',
                     color=self.colors[1], linewidth=2)
      axes[3].set_title(f'{self.model_name} - Recall Curves', fontsize=14, fontweight='bold')
      axes[3].set_xlabel('Epochs')
      axes[3].set_ylabel('Recall')
      axes[3].legend()
      axes[3].grid(True, alpha=0.3)

    # 5. AUC曲线（如果存在）
    if 'auc' in history_dict:
      axes[4].plot(history_dict['auc'], label='Training AUC',
                   color=self.colors[0], linewidth=2)
      if 'val_auc' in history_dict:
        axes[4].plot(history_dict['val_auc'], label='Validation AUC',
                     color=self.colors[1], linewidth=2)
      axes[4].set_title(f'{self.model_name} - AUC Curves', fontsize=14, fontweight='bold')
      axes[4].set_xlabel('Epochs')
      axes[4].set_ylabel('AUC')
      axes[4].legend()
      axes[4].grid(True, alpha=0.3)

    # 6. 学习率曲线（如果存在）
    if 'lr' in history_dict:
      axes[5].plot(history_dict['lr'], label='Learning Rate',
                   color=self.colors[2], linewidth=2)
      axes[5].set_title(f'{self.model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
      axes[5].set_xlabel('Epochs')
      axes[5].set_ylabel('Learning Rate')
      axes[5].set_yscale('log')
      axes[5].legend()
      axes[5].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

  def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵

    参数：
    - y_true: 真实标签
    - y_pred: 预测标签
    - save_path: 保存路径
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Cat', 'Dog'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')

    plt.title(f'{self.model_name} - Confusion Matrix', fontsize=14, fontweight='bold')

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # 打印分类报告
    from sklearn.metrics import classification_report
    print(f"\n{self.model_name} 分类报告:")
    print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

  def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
    """
    绘制ROC曲线

    参数：
    - y_true: 真实标签
    - y_pred_proba: 预测概率
    - save_path: 保存路径
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color=self.colors[0], lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{self.model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return roc_auc

  def plot_prediction_samples(self, X_test, y_true, y_pred, y_pred_proba,
                              n_samples=10, save_path=None):
    """
    绘制预测样本

    参数：
    - X_test: 测试图像
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_pred_proba: 预测概率
    - n_samples: 样本数量
    - save_path: 保存路径
    """
    # 随机选择样本
    indices = random.sample(range(len(X_test)), min(n_samples, len(X_test)))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
      ax = axes[i]

      # 显示图像
      img = X_test[idx]
      if img.max() <= 1.0:
        img = (img * 255).astype('uint8')

      ax.imshow(img)

      # 获取预测信息
      true_label = "Dog" if y_true[idx] == 1 else "Cat"
      pred_label = "Dog" if y_pred[idx] == 1 else "Cat"
      proba = y_pred_proba[idx]

      # 设置标题颜色（正确为绿色，错误为红色）
      color = 'green' if true_label == pred_label else 'red'

      ax.set_title(f'True: {true_label}\nPred: {pred_label}\nProb: {proba:.3f}',
                   color=color, fontsize=10)
      ax.axis('off')

    plt.suptitle(f'{self.model_name} - Prediction Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

  def create_comparison_report(self, models_results, save_path=None):
    """
    创建模型比较报告

    参数：
    - models_results: 字典，包含模型名称和结果
    - save_path: 保存路径
    """
    metrics = ['accuracy', 'precision', 'recall', 'auc', 'loss']

    # 创建数据框
    data = []
    for model_name, results in models_results.items():
      row = {'Model': model_name}
      for metric in metrics:
        if metric in results:
          row[metric.capitalize()] = results[metric]
      data.append(row)

    df = pd.DataFrame(data)

    # 绘制比较条形图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics[:5]):  # 前5个指标
      if metric.capitalize() in df.columns:
        ax = axes[i]
        bars = ax.bar(df['Model'], df[metric.capitalize()],
                      color=self.colors[:len(df)])
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值
        for bar in bars:
          height = bar.get_height()
          ax.text(bar.get_x() + bar.get_width() / 2., height,
                  f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # 第6个子图：显示表格
    axes[5].axis('tight')
    axes[5].axis('off')

    # 创建表格
    table_data = []
    for _, row in df.iterrows():
      table_data.append(row.tolist())

    table = axes[5].table(cellText=table_data,
                          colLabels=df.columns,
                          cellLoc='center',
                          loc='center',
                          colColours=['lightgray'] * len(df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
      plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # 打印数据框
    print("\n模型性能比较:")
    print(df.to_string(index=False))

    return df


# 使用示例
if __name__ == "__main__":
  # 假设我们已经有了训练历史
  # 这里创建模拟数据来演示
  visualizer = TrainingVisualizer(model_name="AlexNet")


  # 创建模拟训练历史
  class MockHistory:
    def __init__(self):
      self.history = {
        'loss': [0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25, 0.23, 0.22, 0.21, 0.2, 0.19],
        'accuracy': [0.65, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95],
        'val_accuracy': [0.7, 0.77, 0.82, 0.86, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93],
        'precision': [0.66, 0.76, 0.81, 0.86, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95],
        'val_precision': [0.71, 0.78, 0.83, 0.87, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94],
        'recall': [0.64, 0.74, 0.79, 0.84, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94],
        'val_recall': [0.69, 0.76, 0.81, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92],
        'auc': [0.68, 0.78, 0.83, 0.88, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97],
        'val_auc': [0.73, 0.79, 0.84, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
      }


  mock_history = MockHistory()
  visualizer.set_history(mock_history)

  # 绘制训练历史
  visualizer.plot_training_history(save_path='./data/visualizations/training_history.png')