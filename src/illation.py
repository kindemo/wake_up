import tensorflow as tf
import os
from illation2 import predict_audio
import numpy as np

# 加载模型
imported = tf.saved_model.load("D:\\PycharmProjects\\wark_by_voice\\saved")


# 遍历文件夹并预测
verify_dir = "D:\\PycharmProjects\\wark_by_voice\\verify"
categories = ["0_non_wake", "1_wake"]
results = {"0_non_wake": [], "1_wake": []}
correct_count = {"0_non_wake": 0, "1_wake": 0}
total_count = {"0_non_wake": 0, "1_wake": 0}

for category in categories:
    category_dir = os.path.join(verify_dir, category)
    for file_name in os.listdir(category_dir):
        file_path = os.path.join(category_dir, file_name)
        if file_path.endswith(".wav"):
            total_count[category] += 1
            class_ids, probabilities = predict_audio(file_path)
            results[category].append((file_name, probabilities))
            if (category == "0_non_wake" and class_ids == 0) or (category == "1_wake" and class_ids == 1):
                correct_count[category] += 1



# 输出每个类别的文件名和预测概率
for category in categories:
    print(f"\nCategory: {category}")
    for file_name, probabilities in results[category]:
        print(f"File: {file_name}, Probabilities: {probabilities}")

# 计算分类正确率
for category in categories:
    accuracy = correct_count[category] / total_count[category] if total_count[category] > 0 else 0
    print(f"\nCategory: {category}, Accuracy: {accuracy:.2f}")

# 计算总分类正确率
total_correct = correct_count["0_non_wake"] + correct_count["1_wake"]
total_files = total_count["0_non_wake"] + total_count["1_wake"]
total_accuracy = total_correct / total_files if total_files > 0 else 0
print(f"\nTotal Accuracy: {total_accuracy:.2f}")

# # 绘制预测的概率分布
# plt.bar('wake_words_probability', prediction_probabilities[0])
# plt.xlabel('Class')
# plt.ylabel('Probability')
# plt.title('Prediction Results')
# plt.show()


# # 预测函数
# def predict_audio(file_path):
#     audio = tf.io.read_file(file_path)
#     audio, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
#     audio = tf.squeeze(audio, axis=-1)
#     audio = audio[tf.newaxis, :]  # 增加批次维度
#     predictions = imported(audio)
#
#     # 检查predictions的结构
#     # print("Predictions structure:", predictions)
#
#     # 提取预测结果
#     # class_ids = predictions['class_ids'].numpy()
#
#
#     # class_names = predictions['class_names'].numpy()
#     # class_names = [name.decode('utf-8') for name in class_names]
#     # print(class_names)
#     prediction_probabilities = predictions['predictions'].numpy()
#     class_ids = 1 if prediction_probabilities[0] > 0.5 else 0
#
#     # 检查输出的形状
#     # print("Class IDs shape:", class_ids.shape)
#     # print("Prediction probabilities shape:", prediction_probabilities.shape)
#
#     # 返回预测结果
#     return class_ids, prediction_probabilities