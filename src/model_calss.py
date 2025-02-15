import tensorflow as tf
from tensorflow.keras import layers
from Spectrum_processing import get_spectrogram, get_mfcc
import numpy as np

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

# 加权二元交叉熵
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = -weights[0] * y_true * tf.math.log(y_pred + 1e-7) - weights[1] * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        return tf.reduce_mean(loss)
    return loss


class ExportModel(tf.Module):
    def __init__(self, model):
        self.model = model

        # 注册方法的签名
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))

    @tf.function
    def calculate_average_db(self, audio_data):
        """
        计算音频数据的分贝值（dB）。

        参数：
        audio_data (tf.Tensor): 形状为 (None, 16000) 的音频张量。

        返回：
        tf.Tensor: 每个样本的分贝值数组 (N,)
        """
        rms = tf.sqrt(tf.reduce_mean(audio_data ** 2, axis=1))
        db = tf.where(rms > 0, 20 * tf.math.log(rms) / tf.math.log(10.0), -np.inf)
        return db

    @tf.function
    def __call__(self, x):
        # 如果输入是字符串（文件路径），则加载并解码音频
        if x.dtype == tf.string:
            x = tf.io.read_file(x)
            x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]  # 增加批次维度

        # 如果输入已经是音频张量，但缺少批次维度，增加批次维度
        if len(x.shape) == 1:
            x = x[tf.newaxis, :]

        # 计算音频的分贝值
        x_wave_db = self.calculate_average_db(x)

        # 获取频谱图
        # x_spectrogram = get_spectrogram(x)
        x_spectrogram = get_mfcc(x)
        print("x_spectrogram.shape:", x_spectrogram.shape)

        # 获取预测结果中概率最高的索引(多分类）
        # class_ids = tf.argmax(result, axis=-1)
        # class_names = tf.gather(label_names, class_ids)

        # 模型预测
        result = self.model(x_spectrogram, training=False)
        result = tf.squeeze(result, axis=-1)  # 转换为一维张量

        # 设置阈值并判断类别
        threshold = 0.5
        m_db = -28
        class_ids = tf.cast(result > threshold, dtype=tf.int32)

        # 如果分贝值大于 -25dB，则将类别 ID 设置为 0
        class_ids = tf.where(x_wave_db > m_db, 0, class_ids)

        # 假设类别名称为 ["negative", "positive"]
        # label_names = tf.constant(["Non_wake_up", "wake_up"])
        # 获取预测类别名称
        # class_names = tf.gather(label_names, class_ids)

        return {'predictions': result, 'class_ids': class_ids}



# 无需返回
# return {'predictions':result,
#         'class_ids': class_ids,
#         'class_names': class_names}