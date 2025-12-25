import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
import os


class DogCatClassifierApp:
    def __init__(self):
        """
        初始化猫狗分类器应用
        """
        st.set_page_config(
            page_title="猫狗大战分类器",
            page_icon="🐱🐶",
            layout="wide"
        )

        # 加载模型（在实际应用中需要先训练好模型）
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        加载预训练模型
        """
        try:
            # 这里假设模型已经训练并保存
            # 在实际应用中，需要先训练模型
            pass
        except Exception as e:
            st.warning(f"模型加载失败: {e}")
            st.info("请先运行训练脚本训练模型")

    def preprocess_image(self, image, target_size=(224, 224)):
        """
        预处理上传的图像

        参数：
        - image: PIL Image对象
        - target_size: 目标尺寸

        返回：
        - 预处理后的图像
        """
        # 转换为RGB（如果必要）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 调整大小
        image = image.resize(target_size)

        # 转换为numpy数组
        img_array = np.array(image)

        # 归一化
        img_array = img_array.astype('float32') / 255.0

        # 添加批次维度
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_with_model(self, model_name, image_array):
        """
        使用指定模型进行预测

        参数：
        - model_name: 模型名称
        - image_array: 图像数组

        返回：
        - 预测结果
        """
        # 在实际应用中，这里会调用加载的模型进行预测
        # 为了演示，我们返回随机结果

        import random

        # 模拟预测
        if model_name == "AlexNet":
            # 模拟AlexNet预测
            prob = random.uniform(0.7, 0.95)
        elif model_name == "CustomCNN":
            # 模拟CustomCNN预测
            prob = random.uniform(0.75, 0.98)
        else:
            prob = random.uniform(0.5, 0.9)

        # 随机决定是猫还是狗
        true_prob = prob if random.random() > 0.3 else 1 - prob
        prediction = 1 if true_prob > 0.5 else 0

        return {
            'probability': true_prob,
            'class': prediction,
            'class_name': '狗 🐶' if prediction == 1 else '猫 🐱',
            'confidence': abs(true_prob - 0.5) * 2  # 置信度
        }

    def run(self):
        """
        运行Streamlit应用
        """
        # 标题
        st.title("🐱 猫狗大战图像分类器 🐶")
        st.markdown("---")

        # 侧边栏
        with st.sidebar:
            st.header("设置")

            # 模型选择
            model_option = st.selectbox(
                "选择分类模型",
                ["AlexNet", "自定义CNN", "两者比较"]
            )

            # 显示模型信息
            st.subheader("模型信息")

            if model_option == "AlexNet":
                st.info("""
                **AlexNet**:
                - 输入尺寸: 227×227
                - 参数量: ~6千万
                - 特点: 经典深度CNN，5个卷积层，3个全连接层
                """)
            elif model_option == "自定义CNN":
                st.info("""
                **自定义CNN**:
                - 输入尺寸: 224×224
                - 参数量: ~2千万
                - 特点: 轻量级设计，带残差连接
                """)
            else:
                st.info("""
                **模型比较**:
                - 同时使用两个模型进行预测
                - 比较预测结果和置信度
                """)

            st.markdown("---")
            st.subheader("关于")
            st.write("""
            这是一个基于深度学习的猫狗图像分类器。

            上传一张猫或狗的图片，模型会预测它是什么动物。

            **数据集**: Kaggle Dogs vs Cats
            **框架**: TensorFlow/Keras
            **界面**: Streamlit
            """)

        # 主界面
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("上传图像")

            # 上传方式选择
            upload_method = st.radio(
                "选择上传方式",
                ["上传文件", "使用示例图像", "拍摄照片"]
            )

            uploaded_image = None

            if upload_method == "上传文件":
                uploaded_file = st.file_uploader(
                    "选择一张猫或狗的图片",
                    type=['jpg', 'jpeg', 'png', 'bmp']
                )

                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    uploaded_image = image

                    # 显示上传的图像
                    st.image(image, caption="上传的图像", use_column_width=True)

            elif upload_method == "使用示例图像":
                example_option = st.selectbox(
                    "选择示例图像",
                    ["猫示例", "狗示例", "混合示例"]
                )

                # 这里可以使用本地示例图像
                # 在实际应用中，需要准备示例图像文件
                st.info("示例图像功能需要在本地有示例图像文件")

                # 模拟显示示例图像
                if example_option == "猫示例":
                    st.image("https://placekitten.com/300/300",
                             caption="示例：猫", use_column_width=True)
                elif example_option == "狗示例":
                    st.image("https://placedog.net/300/300",
                             caption="示例：狗", use_column_width=True)
                else:
                    st.image("https://placekitten.com/300/200",
                             caption="示例：猫", use_column_width=True)
                    st.image("https://placedog.net/300/200",
                             caption="示例：狗", use_column_width=True)

            else:  # 拍摄照片
                st.info("拍摄照片功能需要摄像头支持")
                # 在实际应用中，可以使用st.camera_input()
                # camera_image = st.camera_input("拍摄一张照片")
                # if camera_image is not None:
                #     uploaded_image = Image.open(camera_image)

        with col2:
            st.header("分类结果")

            if uploaded_image is not None or upload_method == "使用示例图像":
                # 显示处理中的消息
                with st.spinner("正在分析图像..."):
                    # 预处理图像
                    if uploaded_image is not None:
                        processed_image = self.preprocess_image(uploaded_image)

                        # 根据选择的模型进行预测
                        if model_option == "AlexNet":
                            result = self.predict_with_model("AlexNet", processed_image)
                            self.display_results([result], ["AlexNet"])

                        elif model_option == "自定义CNN":
                            result = self.predict_with_model("CustomCNN", processed_image)
                            self.display_results([result], ["自定义CNN"])

                        else:  # 两者比较
                            result1 = self.predict_with_model("AlexNet", processed_image)
                            result2 = self.predict_with_model("CustomCNN", processed_image)
                            self.display_results([result1, result2],
                                                 ["AlexNet", "自定义CNN"])

                            # 比较结果
                            self.display_comparison(result1, result2)

                    else:
                        # 使用示例图像的模拟结果
                        st.info("正在使用示例图像进行模拟预测...")

                        if model_option == "AlexNet":
                            # 模拟结果
                            mock_result = {
                                'probability': 0.87,
                                'class': 1,
                                'class_name': '狗 🐶',
                                'confidence': 0.74
                            }
                            self.display_results([mock_result], ["AlexNet"])

                        elif model_option == "自定义CNN":
                            mock_result = {
                                'probability': 0.92,
                                'class': 0,
                                'class_name': '猫 🐱',
                                'confidence': 0.84
                            }
                            self.display_results([mock_result], ["自定义CNN"])

                        else:
                            mock_result1 = {
                                'probability': 0.87,
                                'class': 1,
                                'class_name': '狗 🐶',
                                'confidence': 0.74
                            }
                            mock_result2 = {
                                'probability': 0.92,
                                'class': 0,
                                'class_name': '猫 🐱',
                                'confidence': 0.84
                            }
                            self.display_results([mock_result1, mock_result2],
                                                 ["AlexNet", "自定义CNN"])
                            self.display_comparison(mock_result1, mock_result2)

            else:
                st.info("请上传一张图像或选择示例图像以开始分类")

                # 显示示例
                st.subheader("示例")
                col_ex1, col_ex2 = st.columns(2)
                with col_ex1:
                    st.image("https://placekitten.com/200/200",
                             caption="猫", use_column_width=True)
                with col_ex2:
                    st.image("https://placedog.net/200/200",
                             caption="狗", use_column_width=True)

    def display_results(self, results, model_names):
        """
        显示预测结果

        参数：
        - results: 预测结果列表
        - model_names: 模型名称列表
        """
        for i, (result, model_name) in enumerate(zip(results, model_names)):
            st.subheader(f"{model_name} 预测结果")

            # 创建两列显示结果
            col_pred, col_prob = st.columns(2)

            with col_pred:
                # 显示预测类别
                if result['class'] == 1:
                    st.metric("预测类别", "狗 🐶", delta="")
                    # 显示狗的表情符号
                    st.write("🐕 🐩 🦮 🐕‍🦺")
                else:
                    st.metric("预测类别", "猫 🐱", delta="")
                    # 显示猫的表情符号
                    st.write("🐈 🐈‍⬛ 😺 😸 😹 😻 😼 😽 🙀 😿 😾")

            with col_prob:
                # 显示概率和置信度
                probability = result['probability']
                confidence = result['confidence']

                # 概率进度条
                st.progress(float(probability))
                st.metric("概率", f"{probability:.2%}")

                # 置信度
                if confidence > 0.7:
                    st.success(f"置信度: {confidence:.2%} (高)")
                elif confidence > 0.4:
                    st.warning(f"置信度: {confidence:.2%} (中)")
                else:
                    st.error(f"置信度: {confidence:.2%} (低)")

            # 分隔线
            if i < len(results) - 1:
                st.markdown("---")

    def display_comparison(self, result1, result2):
        """
        显示两个模型的比较结果

        参数：
        - result1: 第一个模型的预测结果
        - result2: 第二个模型的预测结果
        """
        st.markdown("---")
        st.subheader("模型比较")

        # 创建比较数据
        comparison_data = {
            '模型': ['AlexNet', '自定义CNN'],
            '预测类别': [
                result1['class_name'],
                result2['class_name']
            ],
            '概率': [
                result1['probability'],
                result2['probability']
            ],
            '置信度': [
                result1['confidence'],
                result2['confidence']
            ]
        }

        # 显示表格
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.table(df.style.format({
            '概率': '{:.2%}',
            '置信度': '{:.2%}'
        }))

        # 判断是否一致
        if result1['class'] == result2['class']:
            st.success("✅ 两个模型的预测结果一致！")

            # 显示最终预测
            final_class = result1['class_name']
            avg_prob = (result1['probability'] + result2['probability']) / 2
            avg_conf = (result1['confidence'] + result2['confidence']) / 2

            st.balloons()
            st.markdown(f"### 最终预测: **{final_class}**")
            st.markdown(f"平均概率: **{avg_prob:.2%}**")
            st.markdown(f"平均置信度: **{avg_conf:.2%}**")

        else:
            st.warning("⚠️ 两个模型的预测结果不一致！")

            # 建议
            st.info("""
            **建议**:
            1. 检查图像质量
            2. 尝试不同的角度
            3. 使用更清晰的图像
            4. 人工检查图像内容
            """)


# 运行应用
if __name__ == "__main__":
    app = DogCatClassifierApp()
    app.run()