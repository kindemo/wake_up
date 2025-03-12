from keras import Input
from keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Conv1D, GRU, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Attention



class MFCCPreprocess(Layer):
    def __init__(self, n_mfcc=13, frame_length=400, frame_step=160, **kwargs):
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, waveform):
        # 使用TensorFlow操作获取形状维度
        shape = tf.shape(waveform)
        batch_size = shape[0]
        time_steps = shape[1]
        samples_per_frame = 400

        # 合并批次和时间维度
        waveform_flat = tf.reshape(waveform, [-1, samples_per_frame])

        # 计算STFT（保持原始参数）
        stft = tf.signal.stft(
            waveform_flat,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=512
        )
        spectrogram = tf.abs(stft)

        # 创建Mel滤波器（保持与原始参数一致）
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=40,
            num_spectrogram_bins=257,
            sample_rate=16000,
            lower_edge_hertz=100,
            upper_edge_hertz=4000
        )

        # 转换为Mel频谱
        mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, axes=1)
        log_mel = tf.math.log(mel_spectrogram + 1e-6)

        # 计算MFCC
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :self.n_mfcc]

        # 动态计算输出维度
        output_frames = (samples_per_frame - self.frame_length) // self.frame_step + 1
        output_shape = tf.stack([batch_size, time_steps * output_frames, self.n_mfcc])

        return tf.reshape(mfcc, output_shape)


class EnhancedWakeModel(Model):
    def __init__(self, samples_per_frame=400, n_mfcc=13, l2_reg=1e-4):  # 默认值设为1e-4
        super(EnhancedWakeModel, self).__init__()

        # 添加MFCC预处理层
        self.mfcc_preprocess = MFCCPreprocess(
            n_mfcc=13,
            frame_length=400,
            frame_step=160
        )


        # 时频分支（移除池化层，修正正则化器）
        self.freq_conv = Sequential([
            # Input(shape=input_shape),
            # Reshape((-1, 13, 1)),  # 添加通道维度
            Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            # Reshape((-1, input_shape[1] * 64)),         # 输入形状应为 (76, 13, 1)
            tf.keras.layers.Reshape((-1, 13 * 64)),  # 输出形状 (batch, T, 832)
            Dense(64, kernel_regularizer=l2(l2_reg))  # 修正为l2(l2_reg)
        ])

        # 时间分支
        self.time_conv = Sequential([
            Conv1D(64, 3, padding='causal', kernel_regularizer=l2(l2_reg)),
            LayerNormalization(),
            GRU(128, return_sequences=True),
            GRU(64, return_sequences=True)
        ])

        self.cross_attn = Attention(use_scale=True)
        self.classifier = Sequential([
            Dense(64, activation='swish', kernel_regularizer=l2(l2_reg)),  # 修正为l2(l2_reg)
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        # MFCC预处理
        x = self.mfcc_preprocess(x)  # 形状 (B, T, 13)

        # 时频分支处理
        t_input = tf.expand_dims(x, axis=-1)  # 添加通道维度 -> (B, T, 13, 1)
        f = self.freq_conv(t_input)  # 输出形状 (B, T, 64)

        # 时间分支处理
        t = self.time_conv(x)  # 输入形状 (B, T, 13)

        # 注意力机制
        attended = self.cross_attn([f, t])
        pooled = tf.reduce_mean(attended, axis=1)
        return self.classifier(pooled)

# 测试模型
if __name__ == "__main__":
    # 正确初始化参数
    # input_shape = (76, 13, 1)
    # l2_reg_value = 0.01  # 直接使用数值
    #
    # # 创建模型实例
    # model = EnhancedWakeModel(input_shape, l2_reg=l2_reg_value)
    # model.build(input_shape=(None, 76, 13, 1))
    # model.summary()
    #
    # # 验证前向传播
    # test_input = tf.random.normal(shape=(32, 31, 400, 1))
    # output = model(test_input)
    # print("\n测试输出形状:", output.shape)  # 应为 (32, 1)

    # 假设输入为10秒音频，分片为25个时间步，每个分片包含400个采样点
    test_input = tf.random.normal(shape=(2, 25, 400))  # 批次大小为2
    model = EnhancedWakeModel()
    model(test_input)  # 触发动态构建
    model.summary()