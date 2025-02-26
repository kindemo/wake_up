import tensorflow as tf

def get_mfcc(waveform, n_mfcc=13, frame_length=400, frame_step=160, fft_length=256, num_mel_bins=40, lower_frequency=100, upper_frequency=4000):
    """
    提取音频的MFCC特征。
    输入的waveform形状可以是 (时间步, 帧内采样点) 或 (时间步, 帧内采样点, 通道数量)。
    输出的MFCC特征形状为 (时间步, n_mfcc, 通道数量)。
    """
    # 确保输入是 TensorFlow 张量
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # 获取输入形状
    shape = waveform.shape
    if len(shape) == 2:  # 单通道音频
        time_steps, samples_per_frame = shape
        num_channels = 1
    elif len(shape) == 3:  # 多通道音频
        time_steps, samples_per_frame, num_channels = shape
    else:
        raise ValueError("Waveform shape must be either (time_steps, samples_per_frame) or (time_steps, samples_per_frame, num_channels).")

    # 初始化一个列表来存储每个通道的MFCC特征
    mfccs_list = []

    # 对每个通道分别提取MFCC特征
    for channel in range(num_channels):
        if len(shape) == 2:  # 单通道音频
            single_channel = waveform
        else:  # 多通道音频
            single_channel = waveform[:, :, channel]  # 形状为 (时间步, 帧内采样点)

        # 将时间步和帧内采样点展平为一维序列
        single_channel = tf.reshape(single_channel, [-1])  # 形状为 (时间步 * 帧内采样点,)

        # 计算STFT
        stft = tf.signal.stft(
            single_channel,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=tf.signal.hann_window  # 使用汉明窗
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
        mfcc = tf.signal.dct(log_mel_spectrogram, type=2, axis=-1, norm=None)
        mfcc = mfcc[..., :n_mfcc]  # 取前n_mfcc个系数(最大可取40个桶）

        # 将当前通道的MFCC特征添加到列表中
        mfccs_list.append(mfcc)

    # 将所有通道的MFCC特征合并为一个张量
    if num_channels == 1:
        mfccs = tf.expand_dims(mfccs_list[0], axis=-1)  # 单通道音频：形状为 (时间步, n_mfcc, 1)
    else:
        mfccs = tf.stack(mfccs_list, axis=-1)  # 多通道音频：形状为 (时间步, n_mfcc, 通道数量)

    return mfccs


# 创建一个简单的双通道音频信号（随机噪声）
waveform_multi_channel = tf.random.uniform(shape=(10, 400, 2), dtype=tf.float32)  # 100个时间步，每个时间步160个采样点，2个通道

# 提取MFCC特征
mfccs_multi_channel = get_mfcc(waveform_multi_channel)

print("多通道音频MFCC特征形状：", mfccs_multi_channel.shape)
# 输出应为：(时间步, n_mfcc, 通道数量)，例如 (63, 13, 2)



# # 创建一个简单的单通道音频信号（随机噪声）
# waveform_single_channel = tf.random.uniform(shape=(10, 400), dtype=tf.float32)  # 10个时间步，每个时间步400个采样点
#
# # 提取MFCC特征
# mfccs_single_channel = get_mfcc(waveform_single_channel)
#
# print("单通道音频MFCC特征形状：", mfccs_single_channel.shape)
# # 输出应为：(时间步, n_mfcc, 1)，例如 (63, 13, 1)