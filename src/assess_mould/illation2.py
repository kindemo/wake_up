from pathlib import Path

import tensorflow as tf
import os
from src.preprocessing.wave_processing import squeeze as squeezing

# 加载模型
base_path = Path("D:\PycharmProjects\wark_by_voice")
imported = tf.saved_model.load(base_path / "saved")

# 预测函数（直接从模型提出输出概率）
def predict_audio(file_path):
    # 将路径转换为张量形式 (None, 16000)
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1)
    # audio = squeezing(audio)
    audio = audio[tf.newaxis, :]  # 增加批次维度
    predictions = imported(audio)

    # 检查predictions的结构
    # print("Predictions structure:", predictions)

    # class_names = predictions['class_names'].numpy()
    # class_names = [name.decode('utf-8') for name in class_names]
    # print(class_names)

    class_ids = predictions['class_ids'].numpy()  # 提取预测结果
    pred_probabilities = predictions['predictions'].numpy()


    # 检查输出的形状
    # print("Class IDs shape:", class_ids.shape)
    # print("Prediction probabilities shape:", prediction_probabilities.shape)

    # 返回预测结果
    return class_ids, pred_probabilities


# 验证目录
verify_dir = base_path / "verify"
scp_dir = base_path / "we_train/dev/SPEECHDATA"
categories = ["0_non_wake", "1_wake"]
results = {"0_non_wake": [], "1_wake": []}
correct_count = {"0_non_wake": 0, "1_wake": 0}
total_count = {"0_non_wake": 0, "1_wake": 0}
# 选择处理的文件来源
process_source = "verify_dir"  # 可选值："verify_dir" 或 "scp_file"

# SCP文件路径
scp_file_path = base_path / "we_train/dev/SPEECHDATA/dev.scp"
scp_files = []
too_quiet_true = []
too_quiet_false = []

# 读取SCP文件中的路径
if os.path.exists(scp_file_path):
    with open(scp_file_path, 'r') as scp_file:
        scp_files = [line.strip() for line in scp_file.readlines()]


def process_audio_files(f_path, cate, predict_audio_func):
    """
    处理音频文件并进行预测，统计结果。

    参数:
    - f_path: 单个音频文件路径
    - cate: 真实的类别名称（例如 "1_wake" 或 "0_non_wake"）
    - predict_audio_func: 预测音频的函数，返回 (class_ids, probabilities)

    返回值:
    - prob: 模型的预测概率的列表
    """
    if f_path.endswith(".wav"):
        global results, correct_count, total_count
        total_count[cate] += 1
        c_ids, prob = predict_audio_func(f_path)
        results[cate].append((os.path.basename(f_path), prob))

        # 检查预测是否正确，并更新正确预测计数
        if c_ids == 1 and cate == "1_wake":
            correct_count[cate] += 1
        elif c_ids == 0 and cate == "0_non_wake":
            correct_count[cate] += 1

        if c_ids == 0 and prob > 0.5:
            if cate == "0_non_wake":
                too_quiet_false.append(os.path.basename(f_path))
            elif cate == "1_wake":
                too_quiet_true.append(os.path.basename(f_path))
        return prob
    else:
        print(f"文件 {f_path} 不是 WAV 格式，将被忽略。")
        return None


# 遍历文件夹并预测
for category in categories:
    if category == "0_non_wake":
        # 只处理verify_dir中的"0_non_wake"类别
        category_dir = os.path.join(verify_dir, category)
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)
            # 规范化路径
            file_path = os.path.normpath(file_path)
            try:
                assert file_path == os.path.abspath(file_path)  # 确保绝对路径正确
            except Exception as e:
                print(e)
                print(file_path)
                print(os.path.abspath(file_path))
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

print()
print(f'共{len(too_quiet_true)}个本来是唤醒词因为太小声而被认为是非唤醒词{too_quiet_true}')
print(f'共{len(too_quiet_false)}个非唤醒词因为声音太小而被丢弃{too_quiet_false}')

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