import unittest
import numpy as np
import tensorflow as tf


#mfcc流 (时间步, 帧内采样点, [通道数量]) -> (通道数量, 时间步, n_mfcc)
# @tf.function(jit_compile=True)
def get_mfcc(frame_wave, n_mfcc=13, frame_length=400, fft_length=512, num_mel_bins=40,
             lower_frequency=100, upper_frequency=4000):
    """
    提取音频的MFCC特征（优化版）。
    输入形状为 (帧时间步, 帧内采样点) 或 (帧时间步, 帧内采样点, 通道数)。
    输出形状为 (通道数, 帧时间步, n_mfcc)。
    """
    frame_wave = tf.convert_to_tensor(frame_wave, dtype=tf.float32)

    # 添加通道维度（若输入为2D）
    if len(frame_wave.shape) == 2:
        frame_wave = tf.expand_dims(frame_wave, axis=-1)

    # 调整形状为 (通道数, 时间步, 帧长)
    frame_wave = tf.transpose(frame_wave, perm=[2, 0, 1])  # 形状: [C, T, F]
    C, T, F = tf.shape(frame_wave)[0], tf.shape(frame_wave)[1], tf.shape(frame_wave)[2]

    # 合并通道和时间步以批量处理
    frames_flat = tf.reshape(frame_wave, [C * T, F])  # 形状: [C*T, F]

    # 计算FFT（自动填充至fft_length）
    spectrograms = tf.signal.rfft(frames_flat, fft_length=[fft_length])
    spectrograms = tf.abs(spectrograms)  # 幅度谱, 形状: [C*T, fft_length//2 +1]

    # 创建Mel滤波器矩阵（预计算，复用）
    num_spectrogram_bins = fft_length // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=16000,
        lower_edge_hertz=lower_frequency,
        upper_edge_hertz=upper_frequency
    )

    # 转换到Mel频谱并调整形状
    spectrograms = tf.reshape(spectrograms, [C, T, num_spectrogram_bins])
    mel_spectrograms = tf.tensordot(spectrograms, mel_matrix, axes=[[2], [0]])

    # 计算Log-Mel和MFCC
    log_mel = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :n_mfcc]

    return mfccs




class TestGetMFCC(unittest.TestCase):
    def test_single_channel_input(self):
        """测试单通道输入"""
        waveform = np.random.randn(11, 400).astype(np.float32)  # 单通道输入，形状为 (帧时间步, 帧内采样点)
        n_mfcc = 13
        result = get_mfcc(waveform, n_mfcc=n_mfcc)

        # 验证输出形状
        expected_shape = (1, 76, n_mfcc)  # 输出形状应为 (通道数量, 时间步, n_mfcc)
        self.assertEqual(result.shape, tf.TensorShape(expected_shape))

        # 验证输出值是否为实数
        self.assertTrue(tf.reduce_all(tf.math.is_finite(result)))

        # 验证输出值是否为实数
        self.assertFalse(np.iscomplexobj(result), "输出包含复数")
        # 对于静态张量，直接使用 numpy 的方法来检查是否为有限值
        self.assertTrue(np.all(np.isfinite(result.numpy())))

    # def test_multi_channel_input(self):
    #     """测试多通道输入"""
    #     waveform = np.random.randn(100, 400, 2)  # 多通道输入，形状为 (时间步, 帧内采样点, 通道数量)
    #     n_mfcc = 13
    #
    #     result = get_mfcc(waveform, n_mfcc=n_mfcc)
    #
    #     times = int(np.floor((waveform.shape[0] * waveform.shape[1] - 400) / 160)) + 1
    #
    #     # 验证输出形状
    #     expected_shape = tf.TensorShape([2, times, n_mfcc])    # 输出形状应为 (通道数量, 时间步, n_mfcc)
    #     self.assertEqual(result.shape, tf.TensorShape(expected_shape))
    #
    #     # 验证输出值是否为实数
    #     self.assertTrue(tf.reduce_all(tf.math.is_finite(result)))
    #
    # def test_invalid_input_shape(self):
    #     """测试输入形状不正确的情况"""
    #     waveform = np.random.randn(100, 400, 2, 2)  # 四维输入，形状不正确
    #
    #     with self.assertRaises(ValueError):
    #         get_mfcc(waveform)

    # def test_insufficient_input_length(self):
    #     """测试输入长度不足的情况"""
    #     waveform = np.random.randn(1, 399)  # 输入长度不足，无法生成足够的帧
    #
    #     with self.assertRaises(ValueError):
    #         get_mfcc(waveform)

    # def test_output_compliance(self):
    #     """测试输出是否符合预期"""
    #     waveform = np.random.randn(100, 400)  # 单通道输入
    #     n_mfcc = 13
    #
    #     times = int(np.floor((waveform.shape[0] * waveform.shape[1] - 400) / 160)) + 1
    #     print("times:", times)
    #
    #     # 假设 get_mfcc 函数返回的是静态 TensorFlow 张量
    #     result = get_mfcc(waveform, n_mfcc=n_mfcc)
    #
    #
    #
    #     # 验证输出形状
    #     expected_shape = (1, times, n_mfcc)
    #     self.assertEqual(result.shape, tf.TensorShape(expected_shape))
    #
    #     # 验证输出值是否为实数
    #     self.assertFalse(np.iscomplexobj(result), "输出包含复数")
    #     # 对于静态张量，直接使用 numpy 的方法来检查是否为有限值
    #     self.assertTrue(np.all(np.isfinite(result.numpy())))

        # 验证输出的 MFCC 特征是否合理（例如，值范围）
        # 同样使用 numpy 的方法来检查值范围
        # self.assertTrue(np.all(result.numpy() >= 0))


        # waveform = np.random.randn(100, 400)  # 单通道输入
        # n_mfcc = 13
        #
        # result = get_mfcc(waveform, n_mfcc=n_mfcc)
        #
        # # 验证输出形状
        # expected_shape = (1, waveform.shape[0], n_mfcc)
        # self.assertEqual(result.shape, expected_shape)
        #
        # # 验证输出值是否为实数
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(result)))
        #
        # # 验证输出的 MFCC 特征是否合理（例如，值范围）
        # self.assertTrue(tf.reduce_all(result >= 0))

if __name__ == '__main__':
    unittest.main()






# 将音频波形转换为频谱图，通过短时傅里叶变换（STFT）计算频谱图，并取其绝对值，然后添加一个通道维度，使其可以作为卷积层的输入
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]  # 添加通道维度
    # spectrogram = tf.image.resize(spectrogram, (128, 128))  # 统一频谱图的形状
    return spectrogram


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



# # 从音频数据集创建频谱图(对数)数据集
# def make_spec_ds(ds):
#     return ds.map(
#         map_func=lambda audio, label: (get_spectrogram(audio), label),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )

