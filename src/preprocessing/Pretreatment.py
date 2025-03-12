import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import librosa
import scipy.io.wavfile as wavfile
import tensorflow as tf

from src.preprocessing.Augmentation import AudioAugmenter
from src.preprocessing.Spectrum_processing import get_mfcc
from src.preprocessing.windowing import get_windows
from src.preprocessing.wave_processing import squeeze as squeezing


def split_audio_channels(wave, s_rate, frame_length=400, n_mfcc=13, num_win=31):
    """
    分割成通道的核心调用代码
    :param wave: 波形数据需要是TensorFlow 张量
    :param s_rate:
    :param frame_length:
    :param n_mfcc:
    :param num_win:
    :return: 返回形状为 (num_channels, 76, 13) 的张量。
    """
    # 先测试无端点检测
    # 确保输入类型
    data = tf.cast(wave, dtype=tf.float32)
    # 转换为单声道
    data = squeezing(data)
    # 检查是否为单声道
    if len(data.shape) > 1:
        raise ValueError("输入音频必须是单声道！")

    # 将数据归一化到 [-1.0, 1.0]
    waveform = data / (tf.reduce_max(tf.abs(data)) + 1e-6)

    # 进行无重叠拼帧
    windows = get_windows(waveform, frame_length, num_windows=num_win)

    # print(f"windows shape: {windows.shape}")
    # 测试标记
    # channels = get_mfcc(windows, num_windows=num_win)
    # return tf.convert_to_tensor(channels, dtype=tf.float32)
    return tf.convert_to_tensor(windows, dtype=tf.float32)



    # try:
    #     # 提取特征获取帧的多通道，假设返回形状为(num_channels, num_windows, n_mfcc)
    #     channels = get_mfcc(windows, num_windows=num_win)
    #
    #     # 检查 channels 的形状是否正确
    #     channels_shape = tf.shape(channels)
    #     print(f"channels_shape: {channels_shape}")
    #     if len(channels_shape) != 3 or channels_shape[-1] != n_mfcc:
    #         raise ValueError(
    #             f"MFCC features shape is incorrect), "
    #             f"got {channels_shape}")
    #     return tf.convert_to_tensor(channels, dtype=tf.float32)
    # except Exception as e:
    #     print(f"Error getting MFCC features: {e}")
    #     sum_mfcc_frames = int((frame_length * num_win - frame_length) / 160 + 1)
    #     return tf.zeros((1, sum_mfcc_frames, n_mfcc), dtype=tf.float32)


def loading_file2channels(file_path, frame_length=400, n_mfcc=13, num_win=31):
    """
    输入： 文件路径
    按照默认参数会被划分为(400*31-400)+1=76个帧
    将音频张量划分为多个通道，每个通道的形状为 (76, 13)。
    返回形状为 (num_channels, 76, 13) 的张量。
    """
    # 对tensorflow解码
    if isinstance(file_path, bytes):
        file_path = file_path.decode("utf-8")
    elif isinstance(file_path, str):
        pass
    elif isinstance(file_path, tf.Tensor) and file_path.dtype == tf.string:
        file_path = file_path.numpy().decode("utf-8")
    else:
        print(tf.shape(file_path))
        raise TypeError("file_path must be str, bytes, or tf.string Tensor")
    # 加载音频文件
    warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)  # 忽略元数据无法读取的警告
    assert isinstance(file_path, str), "file_path must be str"
    # 用tensorflow自带的库
    audio_binary = tf.io.read_file(file_path)
    wave, s_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    wave = squeezing(wave)
    s_rate = tf.cast(s_rate, dtype=tf.float32)

    # # 加载NOISEX-92数据集路径
    # noise_paths = Path('D:/PycharmProjects/wark_by_voice/noisex-92-master')
    # # 初始化增强器
    # augmenter = AudioAugmenter(
    #     noise_paths=noise_paths,
    #     target_sr=16000
    # )
    # # 启用数据增强
    # augmented = augmenter.augment(wave)
    # # 验证长度保持
    # assert len(augmented) == len(wave)  # True

    return split_audio_channels(wave, s_rate, frame_length, n_mfcc, num_win)



def load_and_split_audio(file_path: str, label: int, frame_length=400, n_mfcc=13, num_win=31):
    """
    加载音频文件并划分通道，返回通道和标签。
    """
    sum_mfcc_frames = int((frame_length * num_win - frame_length) / 160 + 1)

    def py_load_and_split_audio(file_path_str, label_py):
        channels = loading_file2channels(file_path_str, frame_length, n_mfcc, num_win)
        # ！此处不能扩展维度
        num_channels = channels.shape[0]
        labels = tf.repeat(label_py, num_channels)
        # print(f"channels shape: {channels.shape}, labels: {labels}")
        return channels, labels

    channels, labels = tf.numpy_function(
        py_load_and_split_audio,
        [file_path, label],
        [tf.float32, tf.int32]
    )

    # 重点关注
    # channels.set_shape([None, sum_mfcc_frames, n_mfcc])
    channels.set_shape([None, num_win, frame_length])
    labels.set_shape([None])
    return channels, labels


def preprocess_dataset(dataset, frame_length=400, n_mfcc=13, num_win=31):
    """
    预处理数据集，加载并划分音频文件。
    返回: 数据集
    """
    # # 创建初始 Dataset
    # dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    # 使用 map 提取标签
    label_dataset = dataset.map(lambda _, label: label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 应用 load_and_split_audio 函数
    # from_tensor_slices方法下file_path为tf.string使用需要转换
    dataset = dataset.flat_map(lambda file_path, label:
                               tf.data.Dataset.from_tensor_slices(
                                   load_and_split_audio(file_path, label, frame_length, n_mfcc, num_win)
                               )
                               )
    return dataset, label_dataset


class TestLoadAndSplitAudio(unittest.TestCase):
    @patch(__name__ + '.split_audio_channels')
    def test_load_and_split_audio(self, mock_split_audio_channels):
        # 模拟 split_audio_channels 函数的返回值
        mock_channels = tf.random.normal([31, 76, 13])
        mock_split_audio_channels.return_value = mock_channels

        # 定义测试输入

        file_path = str("D:/PycharmProjects/wark_by_voice/verify/0_non_wake/1森林－昆虫－mcx20070416.wav")
        label = tf.constant(0, dtype=tf.int32)

        # 调用被测试函数
        channels, labels = load_and_split_audio(file_path, label, num_win=31)

        # 检查输出类型
        self.assertEqual(channels.dtype, tf.float32)
        self.assertEqual(labels.dtype, tf.int32)

        # 检查输出形状
        self.assertEqual(len(channels.shape), 3)
        self.assertEqual(len(labels.shape), 1)
        self.assertEqual(labels.shape[0], channels.shape[0])


if __name__ == '__main__':
    unittest.main()









# 扩展为四维数据，应用于每个样本的特征部分（x），而标签部分（y）保持不变。
# dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))  # 在最后一个维度扩展

# # 绘制音频波形图和频谱图的函数
# plot_waveform_and_spectrogram(waveform, spectrogram, label, figsize=(12, 8))


# # 取出波形数据
# for example_audio, example_labels in train_ds.take(1):
#     print(f"example_audio.shape: {example_audio.shape}")
#     print(f"example_labels.shape: {example_labels.shape}")
#     # 绘制取出的前九个音频波形
#     plot_audio_waveforms(example_audio, example_labels, label_names, rows=3, cols=3, figsize=(16, 10))
#     break


# # 打印转换后的效果（各向量维度）
# for i in range(3):
#     # 标签：label_names[example_labels[0]]
#     # 波形：example_audio[0]
#     # 频谱图：get_spectrogram(example_audio[0])
#     label = label_names[example_labels[i]]
#     waveform = example_audio[i]
#     spectrogram = get_spectrogram(waveform)
#
#     print('Label:', label)
#     print('Waveform shape:', waveform.shape)
#     print('Spectrogram shape:', spectrogram.shape)
#     print('Audio playback\n')
#     display.display(display.Audio(waveform, rate=16000))
