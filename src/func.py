import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models,Input


# 用于删除音频数据的额外轴，因为音频数据只包含单声道
def squeeze(audio, labels):
    print(f'通道前：{audio.shape}')
    # 如果音频是双声道，取平均值转换为单声道
    # 检查是否为双声道或多声道
    if len(audio.shape) > 2 and audio.shape[-1] is None:  # 如果有多个通道
        audio = tf.reduce_mean(audio, axis=-1)  # 对通道取平均值
    # 删除最后一个维度
    if len(audio.shape) > 2 and audio.shape[-1] == 1:
        audio = tf.squeeze(audio, axis=-1)
    print(f'通道后：{audio.shape}')
    return audio, labels


# 将音频波形转换为频谱图，通过短时傅里叶变换（STFT）计算频谱图，并取其绝对值，然后添加一个通道维度，使其可以作为卷积层的输入
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # 添加通道维度
    # spectrogram = tf.image.resize(spectrogram, (128, 128))  # 统一频谱图的形状
    return spectrogram

# 从音频数据集创建频谱图数据集
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
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
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)

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



# 手动实现残差块
def residual_block(x, filters, kernel_size=3, stride=1, l2_reg=None):
    """
    手动实现一个残差块。
    :param x: 输入张量
    :param filters: 卷积核数量
    :param kernel_size: 卷积核大小
    :param stride: 卷积步长
    :param l2_reg: L2正则化
    :return: 输出张量
    """
    # 主路径
    conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2_reg)(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2_reg)(conv1)
    conv2 = layers.BatchNormalization()(conv2)

    # 跳跃连接
    if stride > 1 or x.shape[-1] != filters:  # 如果步长>1或输入输出通道不一致
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', kernel_regularizer=l2_reg)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = x

    # 将主路径和跳跃连接相加
    output = layers.Add()([conv2, shortcut])
    output = layers.Activation('relu')(output)

    return output