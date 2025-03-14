# data_preprocessing.py
import tensorflow as tf
from tensorflow.keras import layers

def normalize_data(train_ds, val_ds):
    """创建归一化层，并根据训练集数据适应归一化参数"""
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_ds.map(lambda spec, label: spec))
    return norm_layer

def preprocess_data(train_ds, val_ds, norm_layer):
    """将归一化层应用到训练集和验证集"""
    train_ds = train_ds.map(lambda spec, label: (norm_layer(spec), label), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda spec, label: (norm_layer(spec), label), num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds







# def preprocess_dataset(file_paths, labels, batch_size=32):
#     # 创建 TensorFlow 数据集,将文件和标签关联
#     dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
#
#     # 加载音频文件并提取频谱图
#     def load_wav(file_path, label):
#         audio, _ = tf.audio.decode_wav(tf.io.read_file(file_path), desired_channels=1)
#         audio = tf.squeeze(audio, axis=-1)
#         spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
#         spectrogram = tf.abs(spectrogram)
#         return spectrogram, label
#
#     dataset = dataset.map(load_wav, num_parallel_calls=tf.data.AUTOTUNE)
#     dataset = dataset.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#     return dataset