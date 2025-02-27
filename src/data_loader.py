# data_loader.py
import pathlib
from pathlib import Path

import numpy as np


def load_dataset(dataset_path: str = '../AISHELL-WakeUp-1-sample/SPEECHDATA/speech/wav'):
    # 创建数据集路径
    data_dir = pathlib.Path(dataset_path)
    print(f"Loading dataset from: {data_dir}")

    # 获取所有 WAV 文件路径
    file_paths = list(data_dir.glob('*/*.wav'))
    file_paths = [str(path.resolve()) for path in file_paths]  # 转换为绝对路径字符串
    print(f"Found {len(file_paths)} WAV files.")

    # 获取命令（子目录名称）
    commands = np.array([item.name for item in data_dir.iterdir() if item.is_dir() and not item.name.startswith('.')])
    print('Commands:', commands)

    # 标签映射（假设子目录名称代表类别）
    labels = np.array([0 if '0_non_wake' in Path(path).parts[-2] else 1 for path in file_paths])

    return file_paths, labels

    # # 创建 TensorFlow Dataset
    # dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    #
    # # 定义音频解码函数
    # def decode_audio(file_path, label):
    #     audio = tf.io.read_file(file_path)
    #     audio, _ = tf.audio.decode_wav(audio, desired_channels=1)  # 单声道
    #     return audio, label
    #
    # # 应用解码函数
    # dataset = dataset.map(decode_audio, num_parallel_calls=tf.data.AUTOTUNE)
