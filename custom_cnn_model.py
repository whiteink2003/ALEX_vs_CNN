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


class CustomCNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, model_name="CustomCNN"):
        """
        初始化自定义CNN模型

        参数：
        - input_shape: 输入图像形状
        - num_classes: 分类数量
        - model_name: 模型名称
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None

    def build_simple_model(self, use_batch_norm=True, use_dropout=True):
        """
        构建简单版CNN模型（只有3个卷积层）

        参数：
        - use_batch_norm: 是否使用批归一化
        - use_dropout: 是否使用Dropout

        返回：
        - model: 构建的模型
        """
        model = models.Sequential(name=self.model_name + "_Simple")

        # 第一卷积块
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=self.input_shape))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # 第二卷积块
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # 第三卷积块
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # 展平层
        model.add(layers.Flatten())

        # 全连接层（简化）
        model.add(layers.Dense(128, activation='relu'))
        if use_dropout:
            model.add(layers.Dropout(0.5))

        # 输出层
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001, optimizer_name='adam'):
        """
        编译模型

        参数：
        - learning_rate: 学习率
        - optimizer_name: 优化器名称 ('adam', 'sgd', 'rmsprop')

        返回：
        - 编译后的模型
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_model_*()")

        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        # 选择优化器
        if optimizer_name.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        return self.model

    def train_model(self, X_train, y_train, X_val, y_val,
                    epochs=50, batch_size=32, callbacks=None):
        """
        训练模型
        """
        if self.model is None:
            raise ValueError("模型尚未编译，请先调用compile_model()")

        if callbacks is None:
            callbacks = self._get_default_callbacks()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def _get_default_callbacks(self):
        """
        获取默认回调函数
        """
        os.makedirs('./data/models', exist_ok=True)

        model_save_path = f'./data/models/{self.model_name.lower()}_best.h5'

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        return callbacks

    def evaluate_model(self, X_test, y_test):
        """
        评估模型
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        evaluation = self.model.evaluate(X_test, y_test, verbose=0)

        results = {}
        if self.num_classes == 2:
            results = {
                'loss': evaluation[0],
                'accuracy': evaluation[1],
                'precision': evaluation[2],
                'recall': evaluation[3],
                'auc': evaluation[4]
            }
        else:
            results = {
                'loss': evaluation[0],
                'accuracy': evaluation[1]
            }

        return results

    def predict(self, X):
        """
        预测
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        predictions = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            probabilities = predictions.flatten()
            classes = (probabilities > 0.5).astype(int)
            return probabilities, classes
        else:
            probabilities = predictions
            classes = np.argmax(probabilities, axis=1)
            return probabilities, classes

    def save_model(self, filepath=None):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("模型不存在，无法保存")

        if filepath is None:
            filepath = f'./data/models/{self.model_name.lower()}_final.h5'

        self.model.save(filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath):
        """
        加载模型
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")

        # 更新输入形状和类别数量
        self.input_shape = self.model.input_shape[1:]
        self.num_classes = self.model.output_shape[-1]

        return self.model

    def print_model_info(self):
        """
        打印模型信息
        """
        if self.model is None:
            print("模型未构建")
            return

        print(f"模型名称: {self.model.name}")
        print(f"输入形状: {self.input_shape}")
        print(f"输出形状: {self.model.output_shape}")

        # 计算参数数量
        total_params = self.model.count_params()
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        print(f"\n总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"不可训练参数: {non_trainable_params:,}")

        # 估算模型大小
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 假设float32，4字节每个参数
        print(f"模型大小: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    # 创建数据预处理器
    preprocessor = DataPreprocessor(img_size=(224, 224), max_samples=8000)  # 使用小尺寸和少量数据
    X_train, X_val, y_train, y_val = preprocessor.load_and_preprocess_data()

    if X_train is not None:
        print(f"训练数据形状: {X_train.shape}")
        print(f"验证数据形状: {X_val.shape}")

        print(f"\n{'=' * 60}")
        print("训练简单CNN模型 (build_simple_model)")
        print(f"{'=' * 60}")

        try:
            # 创建模型
            custom_cnn = CustomCNNModel(input_shape=X_train.shape[1:],
                                        num_classes=2,
                                        model_name="CustomCNN_Simple")

            # 构建简单模型
            model = custom_cnn.build_simple_model(use_batch_norm=True, use_dropout=True)

            # 编译模型
            custom_cnn.compile_model(learning_rate=0.001, optimizer_name='adam')

            # 打印模型信息
            custom_cnn.print_model_info()

            # 训练模型（只训练2轮用于测试）
            print(f"\n开始训练简单CNN模型...")
            history = custom_cnn.train_model(
                X_train, y_train, X_val, y_val,
                epochs=50,  # 只训练2轮用于测试
                batch_size=32
            )

            # 评估模型
            eval_results = custom_cnn.evaluate_model(X_val, y_val)

            print(f"\n简单CNN模型评估结果:")
            for metric, value in eval_results.items():
                print(f"  {metric}: {value:.4f}")

            # 保存模型
            custom_cnn.save_model(f'./data/models/cnn_simple_end.h5')

        except Exception as e:
            print(f"构建或训练简单模型时出错: {e}")

    else:
        print("数据加载失败，请检查数据集路径和文件格式。")
