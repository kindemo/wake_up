import numpy as np
import tensorflow as tf


# 用于删除音频数据的额外轴，因为音频数据只包含单声道
def squeeze(audio, labels):
    # print(f'通道前：{audio.shape}')
    # 如果音频是双声道，取平均值转换为单声道
    # 检查是否为双声道或多声道
    if len(audio.shape) > 2 and audio.shape[-1] is None:  # 如果有多个通道
        audio = tf.reduce_mean(audio, axis=-1)  # 对通道取平均值
    # 删除最后一个维度
    if len(audio.shape) > 2 and audio.shape[-1] == 1:
        audio = tf.squeeze(audio, axis=-1)
    # print(f'通道后：{audio.shape}')
    return audio, labels


# 计算音频平均分贝
def calculate_average_db(audio_data):
    """
    计算音频数据的分贝值（dB）。

    参数：
    audio_data (np.ndarray ):状为 (None, 16000) 的音频数据数组。

    返回：
    np.ndarray: 每个样本的分贝值数组 (N,)
    """
    # 确保输入是 NumPy 数组
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("audio_data 必须是一个 NumPy 数组。")

    # 确保音频数据的形状正确
    if len(audio_data.shape) != 2 or audio_data.shape[1] != 16000:
        raise ValueError("audio_data 的形状必须是 (None, 16000)。")

    # 计算每个样本的 RMS（均方根值）
    rms = np.sqrt(np.mean(audio_data ** 2, axis=1))

    # 计算分贝（dB）
    db = np.zeros_like(rms)  # 初始化分贝数组
    db[rms > 0] = 20 * np.log10(rms[rms > 0])  # 避免除以零
    db[rms == 0] = -np.inf  # RMS 为零时，分贝值为负无穷

    return db

# def is_wake_word(wav_file_path, threshold_db=-40):
#     """
#     判断音频是否为唤醒词
#     :param wav_file_path: 音频文件路径
#     :param threshold_db: 分贝阈值，低于此值则判断为非唤醒词
#     :return: True（唤醒词）或 False（非唤醒词）
#     """
#     average_db = calculate_average_db(wav_file_path)
#     print(f"音频的平均分贝为：{average_db:.2f} dB")
#     return average_db > threshold_db
#
#
# # 示例使用
# wav_file_path = r"D:\path\to\your\audio.wav"  # 替换为你的音频文件路径
# threshold_db = -40  # 分贝阈值，可以根据实际情况调整
# if is_wake_word(wav_file_path, threshold_db):
#     print("判断为唤醒词")
# else:
#     print("判断为非唤醒词")
