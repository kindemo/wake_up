from keras import Input
from keras.layers import BatchNormalization
from keras.regularizers import l2
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, Reshape
from tensorflow.keras.layers import Conv1D, LayerNormalization, GRU, Dense, Dropout
from tensorflow.keras.layers import Attention
import tensorflow as tf

from src.preprocessing.Spectrum_processing import create_mfcc_model


class EnhancedWakeModel(Model):
    def __init__(self, input_shape, l2_reg=1e-4):  # 默认值设为1e-4
        super(EnhancedWakeModel, self).__init__()

        # 时频分支（移除池化层，修正正则化器）
        self.freq_conv = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)),
            Reshape((-1, input_shape[1] * 64)),         # 输入形状应为 (76, 13, 1)
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
        # mfcc_model = create_mfcc_model()
        f = self.freq_conv(x)               # 输出形状 (B, 76, 64)
        t_input = tf.squeeze(x, axis=-1)    # 去除通道维度，形状 (B, 76, 13)
        t = self.time_conv(t_input)         # 输出形状 (B, 76, 64)
        attended = self.cross_attn([f, t])
        pooled = tf.reduce_mean(attended, axis=1)  # 平均池化
        return self.classifier(pooled)

# 测试模型
if __name__ == "__main__":
    # 正确初始化参数
    input_shape = (76, 13, 1)
    l2_reg_value = 0.01  # 直接使用数值

    # 创建模型实例
    model = EnhancedWakeModel(input_shape, l2_reg=l2_reg_value)
    model.build(input_shape=(None, 76, 13, 1))
    model.summary()

    # 验证前向传播
    test_input = tf.random.normal(shape=(32, 76, 13, 1))
    output = model(test_input)
    print("\n测试输出形状:", output.shape)  # 应为 (32, 1)