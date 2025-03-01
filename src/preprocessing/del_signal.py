# del_signal tf 源代码
import sys
import warnings
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from src.preprocessing.wave_processing import squeeze as squeezing


def short_time_energy(frames):
    """计算短时能量"""
    return tf.reduce_sum(tf.square(frames), axis=1)


def zero_crossing_rate(frames):
    """计算短时过零率"""
    sign_frames = tf.sign(frames)
    diff_sign = tf.abs(sign_frames[:, :-1] - sign_frames[:, 1:])
    return 0.5 * tf.reduce_sum(diff_sign, axis=1)


def endpoint_detection(signal, frame_length, hop_length, energy_threshold, zcr_threshold):
    """
    基于短时能量和过零率的端点检测
    :param signal: 输入音频信号 (Tensor) [-1, 1]归一化数据
    :param frame_length: 帧长 (样本点数)
    :param hop_length: 帧移
    :param energy_threshold: 短时能量阈值
    :param zcr_threshold: 过零率阈值
    :return: 去除两端静音的音频信号
    """
    # 将信号分帧（必须为int64类型)
    frames = tf.signal.frame(signal, frame_length=frame_length, frame_step=hop_length, pad_end=True)
    frame_length = tf.cast(frame_length, tf.int64)
    hop_length = tf.cast(hop_length, tf.int64)

    # 计算每一帧的短时能量和过零率
    energy = short_time_energy(frames)
    zcr = zero_crossing_rate(frames)

    # 第一级判决：基于高能量阈值找到粗略的语音段
    energy_threshold *= tf.reduce_max(energy)
    energy_high = energy > energy_threshold

    # print(f"energy:{energy}, energy_threshold:{energy_threshold}")
    # print(f"帧总长度：{energy.shape}")

    start = tf.argmax(tf.cast(energy_high, tf.int64))  # 找到第一个满足能量阈值的帧
    end = tf.cast(tf.size(energy_high), tf.int64) - tf.argmax(tf.reverse(tf.cast(energy_high, tf.int64), axis=[0]),
                                                              output_type=tf.int64) - 1
    # print(f"满足能量帧范围：{start}->{end}")
    zcr_hold = zcr_threshold * tf.reduce_max(zcr[start:end+1])

    # print(f"energy_high={energy_high}")
    # print(f"zcr:{zcr}. zcr_hold:{zcr_hold}")

    # 第二级判决：从粗略的语音段向两侧扩展，结合低能量和过零率
    # 第二级判决：从粗略的语音段向两侧扩展，结合低能量和过零率
    def expand_start(start):
        condition = start > 0 and (energy[start - 1] > energy_threshold or
                                   (zcr[start - 1] > zcr_hold and energy[start - 1] > 0.2 * energy_threshold))
        return tf.cond(condition, lambda: expand_start(start - 1), lambda: start)

    def expand_end(end):
        energy_size = tf.cast(tf.size(energy), tf.int64)
        condition = end < energy_size - 1 and (energy[end + 1] > energy_threshold or
                                               (zcr[end + 1] > zcr_hold and energy[end + 1] > 0.2 * energy_threshold))
        return tf.cond(condition, lambda: expand_end(end + 1), lambda: end)


    start = expand_start(start)
    end = expand_end(end)

    # 计算语音段的起始和结束样本索引
    start_sample = start * hop_length
    end_sample = end * hop_length + frame_length
    # print(f"扩展后长度范围：{start}->{end}\n")

    # 提取语音段
    trimmed_signal = signal[start_sample:end_sample]
    return trimmed_signal


def del_signal_ini(sr, data):
    """
    需要确保输入为单声道音频
    降噪 + 双门限噪音检测 (左右两端切片)
    输入 波形data, 采样率sr
    输出 (n,)
    """
    # 确保输入类型
    sr = tf.cast(sr, dtype=tf.float32)
    data = tf.cast(data, dtype=tf.float32)

    # 转换为单声道
    data = squeezing(data)

    # 检查是否为单声道
    if len(data.shape) > 1:
        raise ValueError("输入音频必须是单声道！")

    # 消去直流偏移
    data_mean = tf.reduce_mean(data)
    data = data - data_mean
    # 将数据归一化到 [-1.0, 1.0] tf.audio.decode_wav后再归一化
    signal = data / (tf.reduce_max(tf.abs(data)) + 1e-6)

    # FFT 变换
    data_fft = tf.signal.fft(tf.cast(signal, tf.complex64))
    data_fft_abs = tf.abs(data_fft)

    # 噪声阈值（简单估计）
    threshold = data_mean * 0.15

    # 滤波
    data_fft_filtered = tf.where(data_fft_abs < threshold,
                                 tf.cast(0.0, tf.complex64),  # 将0.0转换为复数
                                 data_fft)
    data_filtered = tf.math.real(tf.signal.ifft(data_fft_filtered))  # 使用 tf.math.real 获取实部

    # 执行端点检测
    trimmed_signal = endpoint_detection(
        data_filtered,
        frame_length=tf.cast(sr * 0.025, tf.int32),  # 25ms帧长
        hop_length=tf.cast(sr * 0.010, tf.int32),  # 10ms帧移
        energy_threshold=0.03,  # 能量阈值
        zcr_threshold=0.3  # 过零率阈值
    )
    # *tf.reduce_max(tf.square(data_filtered))
    return trimmed_signal


class TestDelSignalIni(tf.test.TestCase):

    @patch(__name__ + '.endpoint_detection')
    def test_single_channel_input(self, mock_endpoint_detection):
        # 模拟 endpoint_detection 函数的返回值
        mock_result = tf.random.normal([50])
        mock_endpoint_detection.return_value = mock_result

        # 生成单声道测试数据
        sr = 16000
        data = tf.random.normal([10000])

        result = del_signal_ini(sr, data)

        # 检查输出类型
        self.assertIsInstance(result, tf.Tensor)
        # 检查输出是否为一维张量
        self.assertEqual(result.shape.ndims, 1)
        # 检查输出长度是否与模拟返回值长度一致
        self.assertEqual(result.shape[0], mock_result.shape[0])

    @patch(__name__ + '.endpoint_detection')
    def test_multi_channel_input(self, mock_endpoint_detection):
        # 模拟 endpoint_detection 函数的返回值
        mock_result = tf.random.normal([50])
        mock_endpoint_detection.return_value = mock_result

        # 生成多声道测试数据
        sr = 16000
        data = tf.random.normal([10000, 2])

        result = del_signal_ini(sr, data)

        # 检查输出类型
        self.assertIsInstance(result, tf.Tensor)
        # 检查输出是否为一维张量
        self.assertEqual(result.shape.ndims, 1)
        # 检查输出长度是否与模拟返回值长度一致
        self.assertEqual(result.shape[0], mock_result.shape[0])

    def test_endpoint_detection(self):
        # 创建一个测试信号（包含静音段和语音段）
        sr = 16000
        t = tf.range(int(sr * 1), dtype=tf.float32)
        signal = tf.concat([
            tf.zeros(int(sr * 0.5)),  # 0.5秒静音
            tf.sin(2 * np.pi * 500 * t / sr),  # 1秒语音
            tf.zeros(int(sr * 0.5))  # 0.5秒静音
        ], axis=0)

        # 调用 endpoint_detection 函数
        trimmed_signal = endpoint_detection(
            signal,
            frame_length=int(sr * 0.025),  # 25ms帧长
            hop_length=int(sr * 0.010),  # 10ms帧移
            energy_threshold=0.1 * tf.reduce_max(tf.square(signal)),  # 能量阈值
            zcr_threshold=0.25  # 过零率阈值
        )

        # 检查输出信号是否符合预期
        self.assertGreater(len(trimmed_signal), 0)  # 输出信号长度应大于0
        self.assertLess(len(trimmed_signal), len(signal))  # 输出信号应短于原始信号
        self.assertEqual(trimmed_signal.shape.ndims, 1)  # 输出信号应为一维张量


    def test_endpoint_detection_file(self):
        # 创建一个测试信号（包含静音段和语音段）

        # file_path = "D:/PycharmProjects/wark_by_voice/verify/0_non_wake/在木地板上翻滚-YS070515.wav"
        file_path = "D:/PycharmProjects/wark_by_voice/verify/1_wake/c_ya_mid_2_10_2_noisy.wav"

        # 加载音频文件
        assert isinstance(file_path, str), "file_path must be str"
        # 用tensorflow自带的库
        audio_binary = tf.io.read_file(file_path)
        wave, s_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        wave = squeezing(wave)
        print(f"wave:{wave},wave_max{tf.reduce_max(wave)}")

        sr = tf.cast(s_rate, dtype=tf.float32)
        trimmed_signal = del_signal_ini(sr, wave)


        # 检查输出信号是否符合预期
        self.assertGreater(len(trimmed_signal), 0)  # 输出信号长度应大于0
        self.assertLess(len(trimmed_signal), len(wave))  # 输出信号应短于原始信号
        self.assertEqual(trimmed_signal.shape.ndims, 1)  # 输出信号应为一维张量


if __name__ == '__main__':
    tf.test.main()







# 测试用逻辑
# if not tf.reduce_any(energy_high):
#     print(f"energy:{energy}， energy_threshold:{energy_threshold}")
#     print(f"max_energy:{tf.reduce_max(energy)}")
#     if tf.reduce_max(energy) < 1e-6:
#         return signal   # 无语音段直接返回
#     else:
#         raise ValueError("语音段未检测到，请调整能量阈值。")

# if not tf.reduce_any(energy_high):
#     return signal   # 无语音段直接返回