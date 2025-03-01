import pathlib
from pathlib import Path
import numpy as np
import tensorflow as tf


def load_dataset(dataset_path: str):
    data_dir = Path(dataset_path)
    return [
        str(data_dir / '0_non_wake'),
        str(data_dir / '1_wake')
    ], [0, 1]


def create_interleaved_dataset(folder_paths, labels, block_size = 8):
    # 创建文件夹路径和标签的数据集
    folders_ds = tf.data.Dataset.from_tensor_slices((folder_paths, labels))

    # 使用interleave并行处理每个类别文件夹
    dataset = folders_ds.interleave(
        lambda folder_path, label: tf.data.Dataset.list_files(folder_path + '/*.wav').map(lambda x: (x, label)),
        cycle_length=2,
        block_length=block_size,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # 打乱整个数据集
    dataset = dataset.shuffle(buffer_size=10000, seed=seed)
    return dataset