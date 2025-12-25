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



class AlexNetModel:
    def __init__(self, input_shape=(227, 227, 3), num_classes=2):
        """
        初始化AlexNet模型

        参数：
        - input_shape: 输入图像形状 (height, width, channels)
        - num_classes: 分类数量
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self, use_dropout=True, use_l2_reg=False):
        """
        构建AlexNet模型

        参数：
        - use_dropout: 是否使用Dropout
        - use_l2_reg: 是否使用L2正则化

        返回：
        - model: 构建的Keras模型
        """
        model = models.Sequential(name="AlexNet")

        # L2正则化参数
        l2_reg = regularizers.l2(0.0005) if use_l2_reg else None

        # 第一卷积层
        model.add(layers.Conv2D(96, (11, 11), strides=(4, 4),
                                padding='valid', activation='relu',
                                kernel_regularizer=l2_reg,
                                input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # 第二卷积层
        model.add(layers.Conv2D(256, (5, 5), padding='same',
                                activation='relu',
                                kernel_regularizer=l2_reg))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # 第三、四、五卷积层
        model.add(layers.Conv2D(384, (3, 3), padding='same',
                                activation='relu',
                                kernel_regularizer=l2_reg))
        model.add(layers.Conv2D(384, (3, 3), padding='same',
                                activation='relu',
                                kernel_regularizer=l2_reg))
        model.add(layers.Conv2D(256, (3, 3), padding='same',
                                activation='relu',
                                kernel_regularizer=l2_reg))
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))

        # 展平层
        model.add(layers.Flatten())

        # 全连接层
        model.add(layers.Dense(4096, activation='relu',
                               kernel_regularizer=l2_reg))
        if use_dropout:
            model.add(layers.Dropout(0.5))

        model.add(layers.Dense(4096, activation='relu',
                               kernel_regularizer=l2_reg))
        if use_dropout:
            model.add(layers.Dropout(0.5))

        # 输出层
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """
        编译模型

        参数：
        - learning_rate: 学习率

        返回：
        - 编译后的模型
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_model()")

        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

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

        参数：
        - X_train, y_train: 训练数据
        - X_val, y_val: 验证数据
        - epochs: 训练轮次
        - batch_size: 批量大小
        - callbacks: 回调函数列表

        返回：
        - history: 训练历史
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
        获取默认的回调函数
        """
        # 创建保存目录
        os.makedirs('./data/models', exist_ok=True)
        os.makedirs('./data/logs/alexnet', exist_ok=True)

        callbacks = [
            # 早停
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # 模型检查点
            tf.keras.callbacks.ModelCheckpoint(
                filepath='./data/models/alexnet_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # 学习率调度
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard日志 - 禁用直方图以减少内存
            tf.keras.callbacks.TensorBoard(
                log_dir='./data/logs/alexnet',
                histogram_freq=0,  # 改为0，禁用直方图
                write_graph=True,
                write_images=False
            )
        ]

        return callbacks

    def evaluate_model(self, X_test, y_test):
        """
        评估模型

        参数：
        - X_test, y_test: 测试数据

        返回：
        - evaluation: 评估结果字典
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        evaluation = self.model.evaluate(X_test, y_test, verbose=0)

        # 创建结果字典
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

        参数：
        - X: 输入数据

        返回：
        - predictions: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        predictions = self.model.predict(X, verbose=0)

        if self.num_classes == 2:
            # 二分类，返回概率和类别
            probabilities = predictions.flatten()
            classes = (probabilities > 0.5).astype(int)
            return probabilities, classes
        else:
            # 多分类，返回概率和类别
            probabilities = predictions
            classes = np.argmax(probabilities, axis=1)
            return probabilities, classes

    def save_model(self, filepath='./data/models/alexnet_final.h5'):
        """
        保存模型

        参数：
        - filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("模型不存在，无法保存")

        self.model.save(filepath)
        print(f"模型已保存到 {filepath}")

    def load_model(self, filepath='./data/models/alexnet_best.h5'):
        """
        加载模型

        参数：
        - filepath: 模型文件路径
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")

        # 更新输入形状和类别数量
        self.input_shape = self.model.input_shape[1:]
        self.num_classes = self.model.output_shape[-1]

        return self.model


# 使用示例
if __name__ == "__main__":
    # 创建数据预处理器
    preprocessor = DataPreprocessor(img_size=(227, 227))
    X_train, X_val, y_train, y_val = preprocessor.load_and_preprocess_data()

    # 创建AlexNet模型
    alexnet = AlexNetModel(input_shape=(227, 227, 3), num_classes=2)

    # 构建模型
    model = alexnet.build_model(use_dropout=True, use_l2_reg=True)

    # 编译模型
    alexnet.compile_model(learning_rate=0.001)

    # 打印模型摘要
    model.summary()

    # 训练模型
    history = alexnet.train_model(
        X_train, y_train, X_val, y_val,
        epochs=100,  # 可以增加轮次以获得更好性能
        batch_size=32
    )

    # 评估模型
    results = alexnet.evaluate_model(X_val, y_val)
    print("模型评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 保存模型
    alexnet.save_model()