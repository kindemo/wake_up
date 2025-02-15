import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,Input


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

import tensorflow as tf

def get_mfcc(waveform, n_mfcc=13, frame_length=255, frame_step=128, fft_length=256, num_mel_bins=40, lower_frequency=20, upper_frequency=4000):
    # shape (时间步长, n_mfcc, 1)
    # 计算STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )
    spectrogram = tf.abs(stft)

    # 创建梅尔滤波器组
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=16000,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    # 应用梅尔滤波器组
    mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_filterbank.shape[-1:]))

    # 取对数
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # 计算MFCC
    mfccs = tf.signal.dct(log_mel_spectrogram, type=2, axis=-1, norm='ortho')
    mfccs = mfccs[..., :n_mfcc]  # 取前n_mfcc个系数

    # 添加通道维度
    mfccs = mfccs[..., tf.newaxis]

    return mfccs


# 将音频波形转换为频谱图，通过短时傅里叶变换（STFT）计算频谱图，并取其绝对值，然后添加一个通道维度，使其可以作为卷积层的输入
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # 添加通道维度
    # spectrogram = tf.image.resize(spectrogram, (128, 128))  # 统一频谱图的形状
    return spectrogram

# # 从音频数据集创建频谱图(对数)数据集
# def make_spec_ds(ds):
#     return ds.map(
#         map_func=lambda audio, label: (get_spectrogram(audio), label),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )

# 从音频数据集创建Mel频谱图数据集
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_mfcc(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )


# 用于绘制频谱图，将频谱图的频率转换为对数尺度并转置，以便时间轴在 x 轴上表示
def plot_spectrogram(spectrogram, ax):
    """
    用于绘制频谱图，将频谱图的频率转换为对数尺度并转置，以便时间轴在 x 轴上表示。

    参数:
    - spectrogram: 输入的频谱图数据，形状应为 (time_steps, frequency_bins) 或 (1, time_steps, frequency_bins)。
    - ax: matplotlib 的 Axes 对象，用于绘制频谱图。
    """
    # print(f"当前输入的spectrogram的shape{spectrogram.shape}")
    # 确保输入的频谱图的维度为 2 或 3，如果是 3 维则去掉最后一个轴
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)  # 去掉多余的轴

    # 计算对数频谱图
    # log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    log_spec = spectrogram.T        # Mel

    # 获取频谱图的尺寸
    height, width = log_spec.shape

    # 创建时间轴和频率轴的数据
    X = np.linspace(0, width - 1, num=width, dtype=int)
    Y = range(height)

    # 绘制频谱图
    ax.pcolormesh(X, Y, log_spec)

    # 设置轴标签和标题
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Spectrogram')



