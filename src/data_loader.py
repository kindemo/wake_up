from pathlib import Path
import numpy as np
import tensorflow as tf


def load_dataset_train(dataset_path: str):
    data_dir = Path(dataset_path)
    return [
        str(data_dir / '0_non_wake'),
        str(data_dir / '1_wake'),
        str(data_dir / 'soft_wake_0.7'),
        str(data_dir / 'soft_wake_0.8'),
        str(data_dir / 'soft_wake_0.9')
    ], [0.0, 1.0, 0.7, 0.8, 0.9]

def load_dataset_dev(dataset_path: str):
    data_dir = Path(dataset_path)
    return [
        str(data_dir / '0_non_wake'),
        str(data_dir / '1_wake')
    ], [0.0, 1.0]


def create_interleaved_dataset(folder_paths, labels, block_size = 8, is_training = False):
    # 创建文件夹路径和标签的数据集
    folders_ds = tf.data.Dataset.from_tensor_slices((folder_paths, labels))
    if is_training:
        load = 5
    else:
        load = 2

    # 使用interleave并行处理每个类别文件夹
    dataset = folders_ds.interleave(
        lambda folder_path, label: tf.data.Dataset.list_files(folder_path + '/*.wav').map(lambda x: (x, label)),
        cycle_length=load,
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