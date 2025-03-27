
from keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras import Model, Sequential
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dense, Conv1D, GRU, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Attention
# from tensorflow.signal import hamming_window  # 确保导入正确
from tensorflow.python.ops.signal.window_ops import hamming_window


# class FrameLayer(Layer):
#     def __init__(self, frame_length, frame_step, **kwargs):
#         super(FrameLayer, self).__init__(**kwargs)
#         self.frame_length = frame_length
#         self.frame_step = frame_step
#
#     def call(self, inputs):
#         framed = tf.signal.frame(
#             inputs,
#             frame_length=self.frame_length,
#             frame_step=self.frame_step,
#             pad_end=False,
#             axis=1,
#         )
#         return framed
#
#     def compute_output_shape(self, input_shape):
#         # 计算输出形状
#         num_frames = (input_shape[1] - self.frame_length) // self.frame_step + 1
#         return input_shape[0], num_frames, self.frame_length




class MFCCPreprocess(Layer):
    def __init__(self, n_mfcc=13, frame_length=400, frame_step=160, **kwargs):
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, waveform):
        # 确保输入为 (batch, samples)
        if len(waveform.shape) == 3:
            waveform = tf.squeeze(waveform, axis=-1)

        # (batch, 13200)
        # 计算STFT（保持原始参数） fft_length可以设为None 原512
        stft = tf.signal.stft(
            waveform,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            window_fn=hamming_window,  # 使用指定的窗函数
            fft_length=512,
            pad_end=False
        )
        spectrogram = tf.abs(stft)      # 形状: (batch, num_frames, num_bins=257)

        # 创建Mel滤波器（保持与原始参数一致）
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=64,
            num_spectrogram_bins=257,
            sample_rate=16000,
            lower_edge_hertz=50,
            upper_edge_hertz=8000
        )

        # 转换为Mel频谱
        mel_spectrogram = tf.tensordot(spectrogram, mel_matrix, axes=1)
        log_mel = tf.math.log(mel_spectrogram + 1e-6)

        # 计算MFCC
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :self.n_mfcc]
        return mfcc

    def compute_output_shape(self, input_shape):
        # ✅ 动态计算帧数
        num_frames = (input_shape[1] - self.frame_length) // self.frame_step + 1
        return input_shape[0], num_frames, self.n_mfcc


# 含有归一化
class EnhancedWakeModel(Model):
    def __init__(self, n_mfcc=13, l2_reg=1e-4):
        super(EnhancedWakeModel, self).__init__()

        self.mfcc_preprocess = MFCCPreprocess(
            n_mfcc=n_mfcc,
            frame_length=400,
            frame_step=160
        )

        # 定义所有归一化层
        self.bn_input = BatchNormalization()  # 输入归一化层

        # 时频分支
        self.freq_conv = Sequential([
            Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            tf.keras.layers.Reshape((-1, 13 * 64)),
            Dense(64, kernel_regularizer=l2(l2_reg))
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
            Dense(64, activation='swish', kernel_regularizer=l2(l2_reg)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        # 预处理
        x = self.mfcc_preprocess(x)  # 输出形状 (B, 81, 13)
        x = self.bn_input(x)  # 使用预定义的归一化层

        # 时频分支
        t_input = tf.expand_dims(x, axis=-1)
        f = self.freq_conv(t_input)

        # 时间分支
        t = self.time_conv(x)

        # 注意力机制
        attended = self.cross_attn([f, t])
        pooled = tf.reduce_mean(attended, axis=1)
        return self.classifier(pooled)


# 测试模型
if __name__ == "__main__":

    # test_input = tf.random.normal(shape=(5, 13200))  # 批次大小为1
    # model = EnhancedWakeModel()
    # model(test_input)  # 触发动态构建
    # model.summary()

    # 测试模型构建
    model = EnhancedWakeModel()
    model.build(input_shape=(None, 22400))  # 输入维度匹配原始数据
    model.summary()

    # 验证前向传播
    test_input = tf.random.normal((32, 22400))  # 批量大小为32的示例输入
    output = model(test_input)
    print(output.shape)  # 应输出 (32, 1)






# print(f"in MFCC shape: {x.shape}")
# print(f"out MFCC shape: {x.shape}")

# x = self.frameLayer(x)     # 输入形状 (B, 13200),输出（B，32400）
# x = tf.keras.layers.Flatten()(x)  # 展平帧
# print(f"frame shape: {x.shape}")


# 使用TensorFlow操作获取形状维度
# shape = tf.shape(waveform)
# print(f'shape: {shape}')
# batch_size = shape[0]
# time_steps = (shape[1] - 400) // 400 + 1
# samples_per_frame = 400

# 合并批次和时间维度
# waveform_flat = tf.reshape(waveform, [-1, 400])
# print(f'waveform shape: {waveform.shape, waveform}')



# # 第二阶段：分小帧并调整维度
# small_frames = tf.signal.frame(
#     big_frames,
#     frame_length=frame_length,
#     frame_step=frame_step,
#
#     pad_end=False,
#     axis=1
# )
# # 应用窗函数
# window = tf.signal.hamming_window(frame_length, dtype=tf.float32)  # 使用汉明窗
# small_frames = small_frames * window  # 将窗函数应用于每个小帧


# # 动态计算输出维度
# print(f"mfcc_shape: {mfcc.shape}")
# print(f"b,t,mfcc:{[batch_size, time_steps, self.n_mfcc]}")
# output_shape = tf.stack([batch_size, time_steps, self.n_mfcc])
# return tf.reshape(mfcc, output_shape)



# class EnhancedWakeModel(Model):
#     def __init__(self, n_mfcc=13, l2_reg=1e-4):  # 默认值设为1e-4
#         super(EnhancedWakeModel, self).__init__()
#
#         # self.frameLayer = FrameLayer(frame_length=400, frame_step=160)
#
#
#         # 添加MFCC预处理层
#         self.mfcc_preprocess = MFCCPreprocess(
#             n_mfcc=n_mfcc,
#             frame_length=400,
#             frame_step=160
#         )
#
#         # 时频分支（移除池化层，修正正则化器）
#         self.freq_conv = Sequential([
#             Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
#             BatchNormalization(),
#             Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
#             tf.keras.layers.Reshape((-1, 13 * 64)),  # 输出形状 (batch, 832)
#             Dense(64, kernel_regularizer=l2(l2_reg))  # 修正为l2(l2_reg)
#         ])
#
#         # 时间分支
#         self.time_conv = Sequential([
#             Conv1D(64, 3, padding='causal', kernel_regularizer=l2(l2_reg)),
#             LayerNormalization(),
#             GRU(128, return_sequences=True),
#             GRU(64, return_sequences=True)
#         ])
#
#         self.cross_attn = Attention(use_scale=True)
#         self.classifier = Sequential([
#             Dense(64, activation='swish', kernel_regularizer=l2(l2_reg)),  # 修正为l2(l2_reg)
#             Dropout(0.3),
#             Dense(1, activation='sigmoid')
#         ])
#
#     def call(self, x):
#         # MFCC预处理 输入形状 (B, 13200) 输出81帧 (B, 81, 13)
#         x = self.mfcc_preprocess(x)
#
#         # 时频分支处理
#         t_input = tf.expand_dims(x, axis=-1)    # 添加通道维度 -> （B, 81, 13, 1）
#         f = self.freq_conv(t_input)             # 输出形状 (B, 81, 64)
#
#         # 时间分支处理
#         t = self.time_conv(x)  # 输入形状 （B，81， 13）
#
#         # 注意力机制
#         attended = self.cross_attn([f, t])
#         pooled = tf.reduce_mean(attended, axis=1)
#         return self.classifier(pooled)