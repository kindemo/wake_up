from keras import Input
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, Reshape
from tensorflow.keras.layers import Conv1D, LayerNormalization, GRU, Dense, Dropout
from tensorflow.keras.layers import Attention
import tensorflow as tf


class EnhancedWakeModel(Model):
    def __init__(self, input_shape, l2_reg=1e-4):
        super(EnhancedWakeModel, self).__init__()

        original_freq = input_shape[1]
        pooled_freq = (original_freq + 1) // 2

        # 时频分支（添加降维层）
        self.freq_conv = Sequential([
            Input(shape=input_shape),
            Conv2D(16, (3, 3), padding='same'),
            MaxPooling2D((1, 2), padding='same'),
            DepthwiseConv2D(3, depth_multiplier=4, padding='same'),
            Reshape((-1, pooled_freq * 16 * 4)),  # 输出形状：(B,26,448)
            Dense(64)  # 新增：将448维降为64维
        ])

        # 时间分支保持不变
        self.time_conv = Sequential([
            Conv1D(64, 5, padding='causal'),
            LayerNormalization(),
            GRU(64, return_sequences=True)
        ])

        self.cross_attn = Attention(use_scale=True)
        self.classifier = Sequential([
            Dense(64, activation='swish'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        f = self.freq_conv(x)  # 现在形状：(B,26,64)
        t_input = tf.squeeze(x, axis=-1)
        t = self.time_conv(t_input)  # 形状：(B,26,64)

        attended = self.cross_attn([f, t])  # 维度已匹配
        pooled = tf.reduce_mean(attended, axis=1)
        return self.classifier(pooled)



# 测试模型
if __name__ == "__main__":
    # 单元测试代码
    def test_reshape_dimension():
        input_shape = (26, 13, 1)
        model = EnhancedWakeModel(input_shape)

        # 模拟输入
        test_input = tf.random.normal(shape=(32, 26, 13, 1))

        # 前向传播跟踪
        print("输入维度:", test_input.shape)  # (32,26,13,1)

        x = model.freq_conv.layers[0](test_input)  # Conv2D
        print("Conv2D后:", x.shape)  # (32,26,13,16)

        x = model.freq_conv.layers[1](x)  # MaxPooling
        print("MaxPool后:", x.shape)  # (32,26,7,16)

        x = model.freq_conv.layers[2](x)  # DepthwiseConv2D
        print("Depthwise后:", x.shape)  # (32,26,7,64)

        x = model.freq_conv.layers[3](x)  # Reshape
        print("Reshape后:", x.shape)  # (32,26,448)


    test_reshape_dimension()

    model = EnhancedWakeModel((26, 13, 1))
    model.build(input_shape=(None, 26, 13, 1))
    model.summary()

    # 输出应包含：
    # reshape (Reshape)          (None, 26, 448)          0
    # gru (GRU)                   (None, 26, 64)           25088
    # attention (Attention)       (None, 26, 64)           0