import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import json
import h5py
import warnings

warnings.filterwarnings('ignore')


class FixedModelComparator:
    def __init__(self, model_paths, model_names):
        """
        修复版本兼容性的模型比较器
        """
        self.model_paths = model_paths
        self.model_names = model_names
        self.models = []
        self.results = {}

    def load_model_with_fix(self, filepath):
        """
        修复InputLayer兼容性问题后加载模型
        """
        try:
            # 方法1：直接加载（可能成功）
            model = tf.keras.models.load_model(filepath)
            return model
        except Exception as e1:
            print(f"  直接加载失败: {e1}")

            try:
                # 方法2：使用compile=False
                model = tf.keras.models.load_model(filepath, compile=False)

                # 重新编译
                if model.output_shape[-1] == 1:  # 二分类
                    model.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                else:  # 多分类
                    model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )

                return model
            except Exception as e2:
                print(f"  方法2失败: {e2}")

                try:
                    # 方法3：手动加载架构和权重
                    with h5py.File(filepath, 'r') as f:
                        # 获取模型配置
                        model_config = f.attrs.get('model_config')
                        if model_config:
                            model_config = json.loads(model_config.decode('utf-8'))

                            # 修复InputLayer配置
                            def fix_inputlayer_config(config):
                                if isinstance(config, dict):
                                    # 修复当前层
                                    if config.get('class_name') == 'InputLayer':
                                        if 'config' in config and 'batch_shape' in config['config']:
                                            # 将batch_shape改为batch_input_shape
                                            config['config']['batch_input_shape'] = config['config'].pop('batch_shape')

                                    # 递归修复嵌套结构
                                    for key, value in config.items():
                                        if isinstance(value, (dict, list)):
                                            fix_inputlayer_config(value)
                                elif isinstance(config, list):
                                    for item in config:
                                        fix_inputlayer_config(item)

                            fix_inputlayer_config(model_config)

                            # 重新创建模型
                            model = tf.keras.models.model_from_json(json.dumps(model_config))

                            # 加载权重
                            model.load_weights(filepath)

                            # 编译
                            if model.output_shape[-1] == 1:
                                model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy']
                                )
                            else:
                                model.compile(
                                    optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )

                            return model
                        else:
                            raise ValueError("无法从文件中读取模型配置")
                except Exception as e3:
                    print(f"  方法3失败: {e3}")
                    raise

    def load_models(self):
        """加载所有模型（使用修复方法）"""
        print("正在加载模型（使用兼容性修复）...")
        for path, name in zip(self.model_paths, self.model_names):
            try:
                model = self.load_model_with_fix(path)
                self.models.append(model)
                print(f"✓ 已加载模型: {name}")
                print(f"  输入形状: {model.input_shape}")
                print(f"  输出形状: {model.output_shape}")
                print(f"  参数数量: {model.count_params():,}")
            except Exception as e:
                print(f"✗ 加载模型 {name} 失败: {e}")
                self.models.append(None)
        print("-" * 50)

    def compare_all_models(self, X_test, y_test):
        """比较所有模型"""
        if not self.models:
            print("请先加载模型！")
            return

        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            if model is not None:
                print(f"正在评估模型: {name}")

                try:
                    # 进行预测
                    y_pred_prob = model.predict(X_test, verbose=0)

                    # 二分类问题
                    if model.output_shape[-1] == 1:
                        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                        y_pred_proba = y_pred_prob.flatten()
                    else:
                        y_pred = np.argmax(y_pred_prob, axis=1)
                        y_pred_proba = y_pred_prob

                    # 计算评估指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                    # 计算AUC-ROC
                    auc_score = None
                    if len(np.unique(y_test)) == 2:  # 二分类
                        try:
                            auc_score = roc_auc_score(y_test, y_pred_proba)
                        except:
                            auc_score = None

                    # 计算混淆矩阵
                    cm = confusion_matrix(y_test, y_pred)

                    results = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc_score,
                        'confusion_matrix': cm,
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }

                    self.results[name] = results

                    print(f"  Accuracy: {accuracy:.4f}")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1-Score: {f1:.4f}")
                    if auc_score:
                        print(f"  AUC-ROC: {auc_score:.4f}")
                    print("-" * 50)

                except Exception as e:
                    print(f"  评估模型 {name} 时出错: {e}")
                    self.results[name] = None

    # 保留原有的可视化方法...
    def plot_comparison_bar_chart(self):
        """绘制模型性能对比柱状图"""
        if not self.results or all(v is None for v in self.results.values()):
            print("没有有效的评估结果可供可视化！")
            return

        # 过滤掉None结果
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        if not valid_results:
            print("没有有效的评估结果可供可视化！")
            return

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        n_metrics = len(metrics)
        n_models = len(valid_results)

        # 准备数据
        model_names = list(valid_results.keys())
        metric_values = np.zeros((n_models, n_metrics))

        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                metric_values[i, j] = valid_results[model_name][metric]

        # 设置图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = plt.cm.Set3(np.linspace(0, 1, n_models))

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            bars = ax.bar(range(n_models), metric_values[:, idx], color=colors)
            ax.set_title(f'{metric.upper()} 对比', fontsize=14, fontweight='bold')
            ax.set_xlabel('模型', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, linestyle='--')

            # 在柱状图上显示数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.suptitle('模型性能对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存图形
        import os
        os.makedirs('./visualizations', exist_ok=True)
        plt.savefig('./visualizations/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 其他可视化方法保持不变...


# 使用示例
if __name__ == "__main__":
    # 设置模型路径和名称
    model_paths = [
        './data/models/alexnet_best.h5',
        './data/models/cnn_simple_end.h5'
    ]

    model_names = [
        'AlexNet',
        'SimpleCNN'
    ]

    # 创建修复版本的比较器
    comparator = FixedModelComparator(model_paths, model_names)

    # 加载模型
    comparator.load_models()

    # 注意：你需要提供测试数据
    # 这里假设你已经有测试数据X_test, y_test
    # 如果没有，可以使用以下代码生成模拟数据用于演示

    # 生成模拟测试数据（用于演示）
    print("\n生成模拟测试数据用于演示...")
    np.random.seed(42)
    n_samples = 200
    X_test = np.random.randn(n_samples, 128, 128, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, n_samples)

    print(f"模拟测试数据形状: {X_test.shape}")
    print(f"模拟测试标签形状: {y_test.shape}")

    # 比较模型
    comparator.compare_all_models(X_test, y_test)

    # 可视化对比结果
    if comparator.results:
        comparator.plot_comparison_bar_chart()
        print("\n对比完成！图表已保存到 ./visualizations/ 目录")
    else:
        print("\n没有有效的对比结果")