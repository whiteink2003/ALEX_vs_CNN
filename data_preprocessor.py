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

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, data_path='./data', img_size=(224, 224), test_size=0.2, random_state=42, max_samples=8000):
        """
        初始化数据预处理器

        参数：
        - data_path: 数据路径
        - img_size: 图像尺寸 (height, width)
        - test_size: 验证集比例
        - random_state: 随机种子
        - max_samples: 最大样本数 (你的数据集是8000张)
        """
        self.data_path = data_path
        self.train_path = os.path.join(data_path, 'train')
        self.test_path = os.path.join(data_path, 'test1')
        self.img_size = img_size
        self.test_size = test_size
        self.random_state = random_state
        self.max_samples = max_samples  # 根据你的数据集大小设置

    def load_and_preprocess_data(self):
        """
        加载并预处理数据

        返回：
        - X_train, X_val, y_train, y_val: 训练集和验证集
        - class_names: 类别名称
        """
        print("正在加载训练数据...")

        images = []
        labels = []

        # 获取所有训练图片
        train_files = os.listdir(self.train_path)

        # 确保只处理猫狗图片，并按顺序处理
        cat_files = [f for f in train_files if f.startswith('cat.')]
        dog_files = [f for f in train_files if f.startswith('dog.')]

        # 按编号排序
        cat_files.sort(key=lambda x: int(x.split('.')[1]))
        dog_files.sort(key=lambda x: int(x.split('.')[1]))

        # 合并文件列表
        train_files = cat_files + dog_files

        print(f"找到 {len(cat_files)} 张猫图片和 {len(dog_files)} 张狗图片")
        print(f"总训练图片: {len(train_files)} 张")

        for filename in tqdm(train_files[:self.max_samples]):  # 使用max_samples限制
            if filename.startswith('cat'):
                label = 0  # 猫
            elif filename.startswith('dog'):
                label = 1  # 狗
            else:
                continue

            img_path = os.path.join(self.train_path, filename)

            # 读取并预处理图像
            try:
                # 使用OpenCV读取图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 无法读取图片 {filename}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 调整大小
                img = cv2.resize(img, self.img_size)

                # 归一化到[0,1]
                img = img.astype('float32') / 255.0

                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {e}")
                continue

        # 转换为numpy数组
        if len(images) == 0:
            print("错误: 没有加载到任何图片!")
            return None, None, None, None

        X = np.array(images)
        y = np.array(labels)

        print(f"数据加载完成: {X.shape[0]} 张图片")
        print(f"猫的数量: {np.sum(y == 0)}")
        print(f"狗的数量: {np.sum(y == 1)}")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        print(f"训练集: {X_train.shape[0]} 张图片")
        print(f"验证集: {X_val.shape[0]} 张图片")
        print(f"训练集形状: {X_train.shape}")
        print(f"验证集形状: {X_val.shape}")

        # 估算内存使用
        train_memory = X_train.nbytes / (1024 ** 3)  # 转换为GB
        val_memory = X_val.nbytes / (1024 ** 3)  # 转换为GB
        print(f"训练集内存占用: {train_memory:.2f} GB")
        print(f"验证集内存占用: {val_memory:.2f} GB")
        print(f"总内存占用: {train_memory + val_memory:.2f} GB")

        return X_train, X_val, y_train, y_val

    def load_test_data(self, test_size=4000):
        """
        加载测试数据

        参数：
        - test_size: 测试集大小

        返回：
        - X_test: 测试图像数据
        - test_filenames: 测试文件名列表
        """
        print("正在加载测试数据...")

        test_images = []
        test_filenames = []

        # 获取测试图片
        test_files = os.listdir(self.test_path)

        # 按编号排序
        test_files.sort(key=lambda x: int(x.split('.')[0]))

        # 限制测试集大小
        test_files = test_files[:test_size]

        print(f"找到 {len(test_files)} 张测试图片")

        for filename in tqdm(test_files):
            img_path = os.path.join(self.test_path, filename)

            try:
                # 使用OpenCV读取图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 无法读取测试图片 {filename}")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 调整大小
                img = cv2.resize(img, self.img_size)

                # 归一化到[0,1]
                img = img.astype('float32') / 255.0

                test_images.append(img)
                test_filenames.append(filename)
            except Exception as e:
                print(f"处理测试图片 {filename} 时出错: {e}")
                continue

        if len(test_images) == 0:
            print("错误: 没有加载到任何测试图片!")
            return None, None

        X_test = np.array(test_images)

        print(f"测试数据加载完成: {X_test.shape[0]} 张图片")
        print(f"测试集形状: {X_test.shape}")

        return X_test, test_filenames

    def create_data_generators(self, batch_size=32, use_augmentation=True):
        """
        创建数据生成器（支持数据增强）
        注意：由于你的数据集不是按子文件夹组织的，我们需要使用flow_from_dataframe

        参数：
        - batch_size: 批量大小
        - use_augmentation: 是否使用数据增强

        返回：
        - train_generator, val_generator: 数据生成器
        """
        print("创建数据生成器...")

        # 创建数据框
        data = []
        train_files = os.listdir(self.train_path)

        for filename in train_files:
            if filename.startswith('cat'):
                label = 'cats'
            elif filename.startswith('dog'):
                label = 'dogs'
            else:
                continue

            data.append({
                'filename': os.path.join(self.train_path, filename),
                'class': label
            })

        df = pd.DataFrame(data)

        # 划分训练集和验证集
        train_df, val_df = train_test_split(
            df, test_size=self.test_size,
            random_state=self.random_state,
            stratify=df['class']
        )

        print(f"训练集: {len(train_df)} 张图片")
        print(f"验证集: {len(val_df)} 张图片")

        # 训练数据生成器（带数据增强）
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)

        # 验证数据生成器（不增强）
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # 创建数据生成器
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='class',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='filename',
            y_col='class',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_generator, val_generator

    def create_test_generator(self, batch_size=32):
        """
        创建测试数据生成器

        参数：
        - batch_size: 批量大小

        返回：
        - test_generator: 测试数据生成器
        """
        print("创建测试数据生成器...")

        # 创建测试数据框
        test_files = os.listdir(self.test_path)
        test_files.sort(key=lambda x: int(x.split('.')[0]))

        test_data = []
        for filename in test_files[:4000]:  # 限制为4000张
            test_data.append({
                'filename': os.path.join(self.test_path, filename)
            })

        test_df = pd.DataFrame(test_data)

        # 测试数据生成器
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filename',
            y_col=None,  # 没有标签
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False
        )

        return test_generator

    def visualize_data_distribution(self, y, title="训练集类别分布"):
        """
        可视化数据分布

        参数：
        - y: 标签数组
        - title: 图表标题
        """
        plt.figure(figsize=(10, 5))

        # 类别分布
        plt.subplot(1, 2, 1)
        class_counts = pd.Series(y).value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        bars = plt.bar(['Cat (0)', 'Dog (1)'], class_counts.values, color=colors)
        plt.title(f'{title} - 类别数量')
        plt.xlabel('类别')
        plt.ylabel('数量')

        # 在柱状图上添加数量标签
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(count), ha='center', va='bottom', fontsize=12)

        # 饼图
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values, labels=['猫', '狗'],
                colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'{title} - 类别比例')

        plt.tight_layout()

        # 确保目录存在
        os.makedirs('./data/visualizations', exist_ok=True)

        plt.savefig(f'./data/visualizations/{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_sample_images(self, X, y, n_samples=10, title="样本图像示例"):
        """
        可视化样本图像

        参数：
        - X: 图像数据
        - y: 标签
        - n_samples: 显示样本数量
        - title: 图表标题
        """
        plt.figure(figsize=(15, 8))

        # 随机选择样本
        indices = random.sample(range(len(X)), min(n_samples, len(X)))

        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i + 1)
            img = X[idx]

            # 如果是归一化的图像，需要反归一化显示
            if img.max() <= 1.0:
                img = (img * 255).astype('uint8')

            plt.imshow(img)
            plt.title(f'标签: {"狗" if y[idx] == 1 else "猫"} ({y[idx]})')
            plt.axis('off')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # 确保目录存在
        os.makedirs('./data/visualizations', exist_ok=True)

        plt.savefig(f'./data/visualizations/{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建数据预处理器
    # 使用较小的图像尺寸和样本数以避免内存问题
    preprocessor = DataPreprocessor(
        img_size=(128, 128),  # 使用较小的尺寸
        max_samples=8000  # 先加载4000张测试
    )

    print("=" * 50)
    print("1. 加载训练数据")
    print("=" * 50)

    # 加载训练数据
    X_train, X_val, y_train, y_val = preprocessor.load_and_preprocess_data()

    if X_train is not None:
        # 可视化训练数据分布
        preprocessor.visualize_data_distribution(y_train, "训练集类别分布")

        # 可视化样本图像
        preprocessor.visualize_sample_images(X_train, y_train, title="训练集样本图像")

        print("\n" + "=" * 50)
        print("2. 加载测试数据")
        print("=" * 50)

        # 加载测试数据
        X_test, test_filenames = preprocessor.load_test_data(test_size=200)  # 先加载200张测试

        if X_test is not None:
            print(f"测试集加载完成，共 {len(test_filenames)} 张图片")
            print(f"测试图片文件名示例: {test_filenames[:5]}")

            # 注意：测试集没有标签，所以不能可视化分布
            # 但我们可以显示一些测试图片
            print("\n显示测试集样本...")

            # 显示测试集前10张图片
            plt.figure(figsize=(15, 8))
            for i in range(min(10, len(X_test))):
                plt.subplot(2, 5, i + 1)
                img = X_test[i]

                # 反归一化显示
                if img.max() <= 1.0:
                    img = (img * 255).astype('uint8')

                plt.imshow(img)
                plt.title(f'测试图片: {test_filenames[i]}')
                plt.axis('off')

            plt.suptitle("测试集样本图像", fontsize=16)
            plt.tight_layout()
            plt.savefig('./data/visualizations/test_sample_images.png', dpi=300, bbox_inches='tight')
            plt.show()

        print("\n" + "=" * 50)
        print("3. 创建数据生成器")
        print("=" * 50)

        # 创建数据生成器
        try:
            train_generator, val_generator = preprocessor.create_data_generators(
                batch_size=32,
                use_augmentation=True
            )

            print(f"类别索引: {train_generator.class_indices}")
            print(f"训练生成器: {train_generator.samples} 个样本")
            print(f"验证生成器: {val_generator.samples} 个样本")

            # 显示生成器中的一批数据
            print("\n显示生成器中的一批数据...")
            batch_x, batch_y = next(train_generator)
            print(f"批数据形状: {batch_x.shape}")
            print(f"批标签形状: {batch_y.shape}")

            # 可视化生成器中的一批图像
            plt.figure(figsize=(12, 6))
            for i in range(min(8, batch_x.shape[0])):
                plt.subplot(2, 4, i + 1)
                img = batch_x[i]

                # 生成器已经归一化了，所以直接显示
                plt.imshow(img)
                plt.title(f'标签: {batch_y[i]:.2f}')
                plt.axis('off')

            plt.suptitle("数据生成器样本", fontsize=16)
            plt.tight_layout()
            plt.savefig('./data/visualizations/generator_sample.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"创建数据生成器时出错: {e}")
            print("这可能是由于数据集没有正确组织导致的。")

    else:
        print("数据加载失败，请检查数据集路径和文件格式。")
        print(f"训练集路径: {preprocessor.train_path}")
        print(f"测试集路径: {preprocessor.test_path}")

        # 检查目录是否存在
        if not os.path.exists(preprocessor.train_path):
            print(f"错误: 训练集目录不存在: {preprocessor.train_path}")
        else:
            train_files = os.listdir(preprocessor.train_path)
            print(f"训练目录中的文件数量: {len(train_files)}")
            print(f"前5个文件: {train_files[:5]}")

        if not os.path.exists(preprocessor.test_path):
            print(f"错误: 测试集目录不存在: {preprocessor.test_path}")
        else:
            test_files = os.listdir(preprocessor.test_path)
            print(f"测试目录中的文件数量: {len(test_files)}")
            print(f"前5个文件: {test_files[:5]}")