import tensorflow as tf
import os
import numpy as np

# 加载模型
imported = tf.saved_model.load("D:\\PycharmProjects\\wark_by_voice\\saved")


# 预测函数
def predict_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1)
    audio = audio[tf.newaxis, :]  # 增加批次维度
    predictions = imported(audio)

    # 检查predictions的结构
    # print("Predictions structure:", predictions)

    # 提取预测结果
    # class_ids = predictions['class_ids'].numpy()


    # class_names = predictions['class_names'].numpy()
    # class_names = [name.decode('utf-8') for name in class_names]
    # print(class_names)
    prediction_probabilities = predictions['predictions'].numpy()
    class_ids = 1 if prediction_probabilities[0] > 0.5 else 0

    # 检查输出的形状
    # print("Class IDs shape:", class_ids.shape)
    # print("Prediction probabilities shape:", prediction_probabilities.shape)

    # 返回预测结果
    return class_ids, prediction_probabilities


# 验证目录
verify_dir = "D:\\PycharmProjects\\wark_by_voice\\verify"
scp_dir = "D:/PycharmProjects/wark_by_voice/we_train/dev/SPEECHDATA"
categories = ["0_non_wake", "1_wake"]
results = {"0_non_wake": [], "1_wake": []}
correct_count = {"0_non_wake": 0, "1_wake": 0}
total_count = {"0_non_wake": 0, "1_wake": 0}

# SCP文件路径
scp_file_path = "D:/PycharmProjects/wark_by_voice/we_train/dev/SPEECHDATA/dev.scp"
scp_files = []

# 读取SCP文件中的路径
if os.path.exists(scp_file_path):
    with open(scp_file_path, 'r') as scp_file:
        scp_files = [line.strip() for line in scp_file.readlines()]

# 选择处理的文件来源
process_source = "scp_file"  # 可选值："verify_dir" 或 "scp_file"


def process_audio_files(f_path, cate, predict_audio_func):
    """
    处理音频文件并进行预测，统计结果。

    参数:
    - file_paths: 音频文件路径列表
    - cate: 类别名称（例如 "1_wake" 或 "0_non_wake"）
    - predict_audio_func: 预测音频的函数，返回 (class_ids, probabilities)

    返回值:
    - prob: 模型的预测概率的列表
    - correct_count: 正确预测的数量
    - total_count: 总文件数量
    """
    global results, correct_count, total_count
    if f_path.endswith(".wav"):
        total_count[cate] += 1
        c_ids, prob = predict_audio_func(f_path)
        results[cate].append((os.path.basename(f_path), prob))
        if c_ids == 1 and cate == "1_wake":
            correct_count[cate] += 1
        elif c_ids == 0 and cate == "0_non_wake":
            total_count[cate] += 1
        return prob



# 遍历文件夹并预测
for category in categories:
    if category == "0_non_wake":
        # 只处理verify_dir中的"0_non_wake"类别
        category_dir = os.path.join(verify_dir, category)
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            # 规范化路径
            file_path = os.path.normpath(file_path)
            assert file_path == os.path.abspath(file_path)  # 确保绝对路劲正确
            # 计数
            probabilities = process_audio_files(file_path, category, predict_audio)
    elif category == "1_wake":
        if process_source == "verify_dir":
            # 处理verify_dir中的"1_wake"类别
            category_dir = os.path.join(verify_dir, category)   # 进入最小子文件夹
            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)   # 获取单个wav文件路径
                # 规范化路径
                file_path = os.path.normpath(file_path)
                assert file_path == os.path.abspath(file_path)
                probabilities = process_audio_files(file_path, category, predict_audio)

        elif process_source == "scp_file":
            # 处理SCP文件中的路径
            for file_path in scp_files:
                file_path = os.path.join(scp_dir, file_path)
                file_path = os.path.normpath(file_path)
                assert file_path == os.path.abspath(file_path)
                probabilities = process_audio_files(file_path, category, predict_audio)



# 输出每个类别的文件名和预测概率
for cate in categories:
    print(f"\nCategory: {cate}")
    for file_name, probabilities in results[cate]:
        print(f"File: {file_name}, Probabilities: {probabilities}")

# 计算分类正确率
for cate in categories:
    accuracy = correct_count[cate] / total_count[cate] if total_count[cate] > 0 else 0
    print(f"\nCategory: {cate}, Accuracy: {accuracy:.2f}")

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