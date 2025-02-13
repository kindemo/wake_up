import os
import pathlib

import numpy as np
import seaborn as sns
import tensorflow as tf

from IPython import display
from keras.layers import BatchNormalization
from tensorflow.keras import layers, models,Input
from tensorflow.keras.regularizers import l2
from pydub import AudioSegment
from tensorflow.keras import regularizers
from func import *
from draw import *
from absl import logging

# 设置 absl 日志级别为 WARNING
# logging.set_verbosity(logging.WARNING)


import gc
gc.collect()    # 清理不必要的内存

# 动态分配内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12*1024)])
            # 设置GPU内存限制为 14 GB
            print('Using GPU', gpu)
    except RuntimeError as e:
        print(e)


# train_spectrogram_ds = train_spectrogram_ds.map(lambda spec, label: (tf.expand_dims(spec, axis=-1), label))

# 转换为16bit的音频
def convert_to_16bit_wav(input_path, output_path):
    AudioSegment.converter = r"D:\AppData\ffmpeg-2025-01-20-git-504df09c34-essentials_build\bin\ffmpeg.exe"
    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_sample_width(2)  # 转换为 16 位深度
    # audio = tf.audio.decode_wav(audio, desired_channels=1)      # 强制转换为单通道 太卡了这里阻止程序运行
    audio.export(output_path, format="wav")


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# DATASET_PATH = '../mini_speech_commands'
DATASET_PATH = '../AISHELL-WakeUp-1-sample\\SPEECHDATA\\speech\\wav'
data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)

# 返回训练集和验证集
# 每个批次中的样本会按类别均匀分布，确保每个批次中都有不同类别的样本。
# 假设你有一个音频文件，其采样率为 16000 Hz，长度为 2 秒。原始音频文件包含 32000 个样本点。通过设置 output_sequence_length=16000，
# 这个音频文件会被裁剪为 1 秒长的片段，保留前 16000 个样本点。如果音频文件长度不足 1 秒（例如 8000 个样本点），则会被填充到 16000 个样本点。

# 从数据文件夹中加载音频数据集，分为训练集和验证集，设置了批量大小、验证集比例、随机种子、输出序列长度等参数
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=128,      # 每次从数据集中取出 128 个音频样本进行训练或验证
    validation_split=0.2,   # 从整个数据集中随机选取 20% 的数据作为验证集，剩余 80% 的数据作为训练集
    seed=0,
    output_sequence_length=16000,
    class_names=['0_Non_wake', '1_wake_words'],  # 显式指定类别名称
    subset='both'
    )   # 同时加载训练集和验证集

# 列出所有类别标签
label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# 列出所有子目录（即命令类别），并过滤掉一些非命令类别的文件
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)


# 应用于测试集和训练集
# tf.data.AUTOTUNE 用于自动调整并行处理的线程数，以优化性能
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# 再拆分
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# 取出波形数据
for example_audio, example_labels in train_ds.take(1):
    print(f"example_audio.shape: {example_audio.shape}")
    print(f"example_labels.shape: {example_labels.shape}")
    # 绘制取出的前九个音频波形
    plot_audio_waveforms(example_audio, example_labels, label_names, rows=3, cols=3, figsize=(16, 10))
    break

# 打印转换后的效果（各向量维度）
for i in range(3):
    # 标签：label_names[example_labels[0]]
    # 波形：example_audio[0]
    # 频谱图：get_spectrogram(example_audio[0])
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback\n')
    display.display(display.Audio(waveform, rate=16000))

# 绘制音频波形图和频谱图的函数
plot_waveform_and_spectrogram(waveform, spectrogram, label, figsize=(12, 8))

# 创建频谱数据集
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

# 检查数据集的输出形状
print(f'train_da_shape:{train_ds.element_spec}')

# 打印数据集的批次形状
audio_shape = train_ds.element_spec[0].shape
label_shape = train_ds.element_spec[1].shape
print(f"Audio shape: {audio_shape}")
print(f"Label shape: {label_shape}")

# 取出频谱数据
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    # example_audio.shape: (10, 16000)
    print(f"example_spectrograms.shape: {example_spectrograms.shape}")
    print(f"example_spect_labels.shape: {example_spect_labels.shape}")
    # 绘制前九张的频谱图
    plot_spectrograms(example_spectrograms, example_spect_labels, label_names, rows=3, cols=3, figsize=(16, 9))
    break


# 将数据集存入内存之中
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(4096).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)


# 适应归一化层
norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

# 定义全局正则化器
l2_reg = regularizers.L2(l2=0.01)

# # 创建数据增强管道
# data_augmentation = tf.keras.Sequential([
#     layers.RandomRotation(0.2),  # 随机旋转
#     layers.RandomZoom(0.2),  # 随机缩放
#     layers.RandomTranslation(0.1, 0.1),  # 随机平移
#     layers.RandomContrast(0.2)  # 随机对比度调整
# ])
# # 应用数据增强
# train_spectrogram_ds = train_spectrogram_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y),
#     num_parallel_calls=tf.data.AUTOTUNE
# )

# 注意力机制
class SelfAttention(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)

    def call(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attended_values = tf.matmul(attn_weights, v)
        return attended_values

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config


# 使用函数式 API 构建模型
inputs = Input(shape=input_shape)
x = layers.Resizing(128, 128)(inputs)  # 如果输入数据的原始尺寸较小，可以跳过这一步
x = layers.BatchNormalization()(x)
x = layers.Conv2D(16, 3, kernel_regularizer=l2_reg)(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# 残差块
x = residual_block(x, filters=16, kernel_size=3, stride=1, l2_reg=l2_reg)
x = residual_block(x, filters=32, kernel_size=3, stride=2, l2_reg=l2_reg)

x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.3)(x)

# --- 新增GRU层 ---
# 将CNN输出的4D特征图转换为3D时序数据（假设时间步在高度维度）
_, height, width, channels = x.shape  # 动态获取维度
x = layers.Reshape((height, width * channels))(x)  # 转换为 (None, time_steps, features)
# x = layers.GRU(16, return_sequences=False, kernel_regularizer=l2_reg)(x)  # GRU输出最后一步
x = layers.GRU(32, return_sequences=True, kernel_regularizer=l2_reg)(x)  # GRU输出所有时间步

# --- 调用注意力机制 ---
attention = SelfAttention(embed_dim=16)  # 假设注意力机制的嵌入维度为8
x = attention(x)  # 应用注意力机制
# 注意力机制后可以提取最后一步的输出
x = layers.Lambda(lambda x: x[:, -1, :])(x)  # 提取GRU最后一个时间步的输出

# x = layers.Flatten()(x)     # 平展为一维向量
x = layers.Dense(16, kernel_regularizer=l2_reg)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# 构建模型
model = models.Model(inputs=inputs, outputs=outputs)
model.summary()

# 加权二元交叉熵
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = -weights[0] * y_true * tf.math.log(y_pred + 1e-7) - weights[1] * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        return tf.reduce_mean(loss)
    return loss

# 加权w使模型更加关注正类
w = [1.1, 1.0]


# loss='binary_crossentropy',
# Adam 优化器
model.compile(
    optimizer='adam',
    loss=weighted_binary_crossentropy(w),
    metrics=['accuracy'],
)


# 设置早停准确率和改善限度
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=2, train_accuracy_threshold=0.8):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.train_accuracy_threshold = train_accuracy_threshold
        self.best_weights = None
        self.best_weights_path = '../temp_weights/best_weights.h5'
        self.best = None
        self.wait = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = float('inf')  # 假设监控的是损失，如果是准确率则初始化为 -inf

    def on_epoch_end(self, epoch, logs=None):
        # 获取验证集损失和训练集准确率
        val_loss = logs.get('val_loss', float('inf'))   # 如果没有 val_loss，则使用一个很大的值
        train_accuracy = logs.get('accuracy', 0.0)  # 或者是 'acc'，取决于你的模型定义

        # 检查训练集准确率是否达到阈值
        if train_accuracy < self.train_accuracy_threshold:
            print(f"\t训练集准确率未达到 {self.train_accuracy_threshold * 100}%，继续训练...")
            return

        # 检查验证集损失是否改善
        if val_loss < self.best:
            self.best = val_loss
            self.wait = 0
            self.model.save_weights(self.best_weights_path)  # 保存权重到磁盘
            # self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\t验证集损失在连续 {self.patience} 个轮次内没有改善，训练提前停止。")
                # self.model.set_weights(self.best_weights)  # 恢复最佳权重
                self.model.load_weights(self.best_weights_path)  # 从磁盘加载权重

# 使用自定义回调函数
EPOCHS = 10
# callbacks：回调函数，当验证集上的损失在连续 2 个轮数内没有改善时，提前停止训练。
callbacks = [
    CustomEarlyStopping(patience=2, train_accuracy_threshold=0.85),
    tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1, update_freq='epoch')
]

history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    callbacks=callbacks,
    verbose=1,
    use_multiprocessing=True,
    workers=4
)

# 绘制损失与准确率曲线
# history 属性是一个字典，记录了训练过程中的各种指标，如损失和准确率
plot_training_history(history, figsize=(16, 6))


# 绘制混淆矩阵

model.evaluate(test_spectrogram_ds, return_dict=True)
y_pred = model.predict(test_spectrogram_ds)
# y_pred = tf.argmax(y_pred, axis=1)    # 多分类时用
y_pred = tf.cast(y_pred >= 0.5, tf.int32).numpy().flatten()
# 真实标签
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
print("True labels:", y_true)
print("Predicted labels:", y_pred)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# 小验证
x = 'D:\\PycharmProjects\\wark_by_voice\\verify\\1_wake\\c_ya_fast_2_10_1_quiet.wav'

# 将输入转换为16bit的音频
convert_to_16bit_wav(x, x)

x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis,...]   # 转化为批次的形式

prediction = model(x)
print("prediction:", prediction)

x_labels = ['wake_words_probability']
wake_word_probability = prediction.numpy()[0][0]  # 提取概率值
plt.bar(x_labels, [wake_word_probability])

plt.title('miya')
plt.ylabel('Probability')
plt.show()

display.display(display.Audio(waveform, rate=16000))



#导出模型
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    # 获取频谱
    x = get_spectrogram(x)
    result = self.model(x, training=False)

    # 获取预测结果中概率最高的索引
    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions':result,
            'class_ids': class_ids,
            'class_names': class_names}


export = ExportModel(model)

try:
    export(tf.constant(str(data_dir/'1_wake_words/SV0001_2_05_F0909.wav')))
except Exception as e:
    print(e)
finally:
    tf.saved_model.save(export, "D:\\PycharmProjects\\wark_by_voice\\saved")
    imported = tf.saved_model.load("D:\\PycharmProjects\\wark_by_voice\\saved")
    imported(waveform[tf.newaxis, :])
    print("end")

