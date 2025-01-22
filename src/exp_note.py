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