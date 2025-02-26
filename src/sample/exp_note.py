# model = models.Sequential([
#     layers.Input(shape=input_shape),
#     layers.Resizing(32, 32),  # 如果输入数据的原始尺寸较小，可以跳过这一步
#     norm_layer,
#     layers.Conv2D(64, 5, kernel_regularizer=l2_reg),
#     BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.3),
#
#     # 添加第一个残差块
#     residual_block(filters=64, kernel_size=3, stride=1, l2_reg=l2_reg),
#
#     # 添加第二个残差块
#     residual_block(filters=128, kernel_size=3, stride=2, l2_reg=l2_reg),
#
#     layers.Conv2D(128, 3, kernel_regularizer=l2_reg),
#     BatchNormalization(),
#     layers.Activation('relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.3),
#     layers.Flatten(),
#     layers.Dense(128, kernel_regularizer=l2_reg),
#     BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_labels)
# ])
#
# model.summary()


# 导出模型
# class ExportModel(tf.Module):
#     def __init__(self, model):
#         self.model = model
#
#         # Accept either a string-filename or a batch of waveforms.
#         # YOu could add additional signatures for a single wave, or a ragged-batch.
#         self.__call__.get_concrete_function(
#             x=tf.TensorSpec(shape=(), dtype=tf.string))
#         self.__call__.get_concrete_function(
#             x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
#
#     @tf.function
#     def calculate_average_db(self, audio_data):
#         """
#         计算音频数据的分贝值（dB）。
#
#         参数：
#         audio_data (tf.Tensor): 形状为 (None, 16000) 的音频张量。
#
#         返回：
#         tf.Tensor: 每个样本的分贝值数组 (N,)
#         """
#         # 计算每个样本的 RMS（均方根值）
#         rms = tf.sqrt(tf.reduce_mean(audio_data ** 2, axis=1))
#         # 计算分贝（dB）
#         db = tf.where(rms > 0, 20 * tf.math.log(rms) / tf.math.log(10.0), -np.inf)
#
#         return db
#
#     @tf.function
#     def __call__(self, x):
#         # If they pass a string, load the file and decode it.
#         if x.dtype == tf.string:
#             x = tf.io.read_file(x)
#             x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000, )
#             x = tf.squeeze(x, axis=-1)
#             x = x[tf.newaxis, :]
#
#         # 获取频谱
#         x_wave_db = self.calculate_average_db(x)
#
#         x = get_spectrogram(x)
#         print("x.shape", x.shape)
#         result = self.model(x, training=False)
#         result = tf.squeeze(result, axis=-1)  # 转换为一维张量
#
#         # 设置阈值
#         threshold = 0.5
#         # 根据阈值判断类别
#         class_ids = tf.cast(result > threshold, dtype=tf.int32)
#
#         # 检查 db 的形状是否与 class_ids 相同
#         if x_wave_db.shape != class_ids.shape:
#             raise ValueError("db 和 class_ids 的形状必须相同。")
#
#         # 如果 db 的值大于 -40 分贝，则将对应的 class_ids 设置为 0
#         class_ids = tf.where(x_wave_db > -40, 0, class_ids)
#
#         return {'predictions': result, 'class_ids': class_ids}