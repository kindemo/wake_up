# import tensorflow as tf
# from pathlib import Path
# from src.preprocessing.wave_processing import squeeze as squeezing
#
# model_path = "D:/PycharmProjects/wark_by_voice/saved"
# loaded_model = tf.saved_model.load(model_path)
#
# @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
# def serving_default(input_audio_path):
#     # 加载音频文件
#     audio_data = tf.io.read_file(input_audio_path)
#     waveform, sample_rate = tf.audio.decode_wav(audio_data)
#     waveform = squeezing(waveform, axis=-1)  # 去掉声道维度
#
#     # 添加 batch 维度
#     waveform = tf.expand_dims(waveform, axis=0)
#
#     try:
#         # 调用模型
#         model_output = loaded_model(waveform)
#     except Exception as e:
#         print(e)
#
#     # 检查返回值类型
#     if isinstance(model_output, dict):
#         pred = model_output['predictions']  # 假设返回值是一个字典，且键为 'predictions'
#         classes = model_output['class_ids']
#     else:
#         pred = model_output  # 如果返回值直接是张量
#         # 计算类别 ID
#         threshold = 0.5
#         classes = tf.cast(pred > threshold, dtype=tf.int64)  # 形状为 (num_channels, num_classes)
#     classes = tf.squeeze(classes, axis=1)  # 将 [1, 6] 转换为 [6]
#     return {"predictions": pred, "class_ids": classes}
#
# # 准备输入数据
# data_dir = Path("D:/PycharmProjects/wark_by_voice/verify")
# audio_file_path = str(Path(data_dir) / '1_wake/c_ya_slow_2_10_3_quiet.wav')
# input_data = tf.constant(audio_file_path, dtype=tf.string)
#
# # 调用模型
# output = serving_default(input_data)
# predictions = output['predictions']
# class_ids = output['class_ids']
#
# # 生成时间轴张量
# time_step = 0.125
# time_axis = tf.range(0, tf.cast(tf.size(class_ids), tf.float32) * time_step, time_step)
#
# # 将 class_ids 和时间轴堆叠成一个二维张量
# # 确保 class_ids 是浮点类型，以便与时间轴对齐
# class_ids = tf.cast(class_ids, tf.float32)
# stacked_tensor = tf.stack([class_ids, time_axis], axis=1)
#
# print("Predictions:", predictions)
# print("Class IDs:", class_ids)
# print("Stacked Tensor:\n", stacked_tensor)
#
import numpy as np
import tensorflow as tf
from pathlib import Path

from matplotlib import pyplot as plt

# 加载导出的模型（包含预处理）
model_path = "D:/PycharmProjects/wark_by_voice/saved"
loaded_model = tf.saved_model.load(model_path)


# # ***
# # 调用文件路径签名
# loaded_model.signatures["file_input"](x=tf.constant("audio.wav"))

# # 打印模型的签名
# print(list(loaded_model.signatures.keys()))  # 查看签名名称
# infer = loaded_model.signatures["serving_default"]
# print(infer.structured_input_signature)  # 查看输入签名
# print(infer.structured_outputs)  # 查看输出签名

# 准备测试音频路径
# data_dir = Path("D:/PycharmProjects/wark_by_voice/verify/1_wake")
data_dir = Path("D:/PycharmProjects/wark_by_voice/train_sample/0_non_wake")
audio_file_path = str(data_dir / '1森林－昆虫－mcx20070416.wav')

# print(list(loaded_model.signatures.keys()))  # 通常为 "serving_default"
# infer = loaded_model.signatures["serving_default"]
# print(infer.inputs)  # 查看输入张量要求


# 直接调用模型处理输入（传入文件路径）
input_data = tf.constant(audio_file_path, dtype=tf.string)
output = loaded_model(input_data)

# 获取输出结果
predictions = output['predictions'].numpy()
class_ids = predictions >= 0.5
print(f'class_ids.shape = {class_ids.shape}')

# 生成时间轴
time_step = 0.4  # 根据每个窗口的持续时间调整
time_axis = [i * time_step for i in range(predictions.shape[0])]

# 组合结果


stacked_result = list(zip(time_axis, class_ids))

# 绘制图像
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(time_axis, predictions, label='Predictions', color='blue')  # 绘制预测结果曲线
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (y=0.5)')  # 在 y=0.5 处画一条红色虚线

# 添加图表标题和标签
plt.title('Model Predictions Over Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示图像




# print("Predictions:\n", predictions)
# # print("Class IDs per window:\n", class_ids)
# # print("Time and Class IDs:\n", stacked_result)
# # 格式化输出
# print("Time\tClass ID")
# print("-" * 20)
# for time, class_id in stacked_result:
#     # 将 numpy 数组转换为普通的 Python 列表并取第一个元素
#     class_id = class_id.tolist()[0]
#     print(f"{time:.3f}\t{class_id}")