from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow as tf

class SelfAttention(layers.Layer):
    """
    Self-attention layer 注意力机制层
    """
    def __init__(self, embed_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.query = layers.Dense(embed_dim)
        self.key = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)

    def call(self, inputs, training=None, mask=None):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        if mask is not None:
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / tf.reduce_sum(attn_weights, axis=-1, keepdims=True)

        attended_values = tf.matmul(attn_weights, v)
        return attended_values

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class ResidualBlock(layers.Layer):
    """
    Residual Block 残差神经网络层
    """
    def __init__(self, filters, kernel_size=3, stride=1, l2_reg=None, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.l2_reg = l2_reg

        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2_reg
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding='same', kernel_regularizer=l2_reg
        )
        self.bn2 = layers.BatchNormalization()

        self.shortcut = None
        self.bn_shortcut = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if input_channels != self.filters or self.stride != 1:
            self.shortcut = layers.Conv2D(
                self.filters, 1, strides=self.stride, padding='same', kernel_regularizer=self.l2_reg
            )
            self.bn_shortcut = layers.BatchNormalization()
        else:
            self.shortcut = layers.Lambda(lambda x: x)
            self.bn_shortcut = layers.Lambda(lambda x: x)
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "l2_reg": self.l2_reg
        })
        return config


class CustomModel(tf.keras.Model):
    """
    模型的主要结构
    """
    def __init__(self, input_shape, num_labels, l2_reg):
        super().__init__()
        self.l2_reg = l2_reg

        self.input_layer = layers.InputLayer(input_shape=input_shape)
        self.bn = layers.BatchNormalization()
        # 保持时间维度
        self.conv = layers.Conv2D(16, 3, padding='same',kernel_regularizer=l2_reg)
        self.max_pool_1_time = layers.MaxPooling2D(pool_size=(2, 1))  # 仅压缩时间维度26->13
        self.res_block1 = ResidualBlock(filters=32, kernel_size=3, stride=1, l2_reg=l2_reg)
        self.dropout = layers.Dropout(0.3)
        self.res_block2 = ResidualBlock(filters=64, kernel_size=3, stride=1, l2_reg=l2_reg)
        # self.freq_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=2)) # 沿频率维度池化 → (batch, 13, 1)

        self.gru = layers.GRU(64, return_sequences=True, kernel_regularizer=l2_reg)
        self.attention = SelfAttention(embed_dim=64)
        self.dense1 = layers.Dense(16, kernel_regularizer=l2_reg)
        self.fin_dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(1, activation='sigmoid')


    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.bn(x)
        x = self.conv(x)
        x = self.max_pool_1_time(x)
        x = self.res_block1(x)
        x = self.dropout(x)
        x = self.res_block2(x)

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, tf.shape(x)[-1]])  # 更通用的reshape方式

        x = self.gru(x)
        x = self.attention(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = self.dense1(x)
        x = self.fin_dropout(x)
        x = self.dense2(x)
        return x


# 测试模型
if __name__ == "__main__":
    input_shape = (32, 32, 3)
    num_labels = 10
    l2_reg = regularizers.L2(0.01)

    model = CustomModel(input_shape=input_shape, num_labels=num_labels, l2_reg=l2_reg)
    model.build(input_shape=(None, *input_shape))
    model.summary()