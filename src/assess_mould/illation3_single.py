# import tensorflow as tf
# from pathlib import Path
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


import tensorflow as tf
from pathlib import Path

# 加载导出的模型（包含预处理）
model_path = "D:/PycharmProjects/wark_by_voice/saved"
loaded_model = tf.saved_model.load(model_path)

# 准备测试音频路径
data_dir = Path("D:/PycharmProjects/wark_by_voice/verify")
# audio_file_path = str(data_dir / '1_wake/c_ya_slow_2_10_3_quiet.wav')
audio_file_path = str(data_dir / '0_non_wake/22 研究工作实验室.wav')

# 直接调用模型处理输入（传入文件路径）
input_data = tf.constant(audio_file_path, dtype=tf.string)
output = loaded_model(input_data)

# 获取输出结果
predictions = output['predictions'].numpy()
class_ids = output['class_ids'].numpy()

# 生成时间轴
time_step = 0.125  # 根据每个窗口的持续时间调整
time_axis = [i * time_step for i in range(class_ids.shape[0])]

# 组合结果
stacked_result = list(zip(time_axis, class_ids))

print("Predictions:\n", predictions)
# print("Class IDs per window:\n", class_ids)
# print("Time and Class IDs:\n", stacked_result)
# 格式化输出
print("Time\tClass ID")
print("-" * 20)
for time, class_id in stacked_result:
    # 将 numpy 数组转换为普通的 Python 列表并取第一个元素
    class_id = class_id.tolist()[0]
    print(f"{time:.3f}\t{class_id}")