import unittest
import numpy as np
import tensorflow as tf


#mfcc流 (时间步, 帧内采样点, [通道数量]) -> (通道数量, 时间步, n_mfcc)
# @tf.function(jit_compile=True)
def get_mfcc(frame_wave, n_mfcc=13, frame_length=400, frame_step=160, num_windows=11, fft_length=512, num_mel_bins=40,
             lower_frequency=100, upper_frequency=4000):
    """
        提取音频的MFCC特征。
        因为已经通过windowing进行了分帧，所以对于每个通道都是num_windows*frame_length的大小，
        即总的采样点固定了，所以每个通道的分帧模式和生成帧数量都相等
        输入的waveform形状可以是 (帧时间步, 帧内采样点) 或 (帧时间步, 帧内采样点, 通道数量)。
        输出的MFCC特征形状为 (通道数量, 小帧时间步, n_mfcc)
    """
    frame_wave = tf.convert_to_tensor(frame_wave, dtype=tf.float32)
    shape = tf.shape(frame_wave)

    # 确定通道数
    if len(frame_wave.shape) == 2:
        time_steps, samples_per_frame = shape[0], shape[1]
        num_channels = 1
        frame_wave = tf.expand_dims(frame_wave, axis=-1)  # 添加通道维度
    else:
        time_steps, samples_per_frame, num_channels = shape[0], shape[1], shape[2]

    # 使用TensorArray代替Python列表
    mfccs_ta = tf.TensorArray(size=num_channels, dtype=tf.float32)

    # 循环处理每个通道
    for channel in tf.range(num_channels):
        single_channel = frame_wave[..., channel]
        single_channel = tf.reshape(single_channel, [-1])  # 展平

        # 计算STFT
        stft = tf.signal.stft(
            single_channel,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )
        spectrogram = tf.abs(stft)

        # 创建Mel滤波器矩阵
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=fft_length // 2 + 1,
            sample_rate=16000,  # 根据实际情况调整
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

        # 转换到Mel频谱
        mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, 1)
        log_mel = tf.math.log(mel_spectrogram + 1e-6)

        # 计算MFCC
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
        mfcc = mfcc[..., :n_mfcc]  # 截取指定系数

        # 调整形状并写入TensorArray
        expected_frames = (tf.size(single_channel) - frame_length) // frame_step + 1
        mfcc = tf.reshape(mfcc, [expected_frames, n_mfcc])
        mfccs_ta = mfccs_ta.write(channel, mfcc)

    # 堆叠所有通道的结果
    mfccs = mfccs_ta.stack()
    return mfccs




class TestGetMFCC(unittest.TestCase):
    def test_single_channel_input(self):
        """测试单通道输入"""
        waveform = np.random.randn(11, 400).astype(np.float32)  # 单通道输入，形状为 (帧时间步, 帧内采样点)
        n_mfcc = 13
        result = get_mfcc(waveform, n_mfcc=n_mfcc)

        # 验证输出形状
        expected_shape = (1, 26, n_mfcc)  # 输出形状应为 (通道数量, 时间步, n_mfcc)
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

