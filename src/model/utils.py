import sys

import tensorflow as tf

# 加权二元交叉熵
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = -weights[0] * y_true * tf.math.log(y_pred + 1e-7) - weights[1] * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        return tf.reduce_mean(loss)
    return loss


class FocalLoss(tf.keras.losses.Loss):
    """支持软标签的Focal Loss，适配包括0.7在内的概率标签"""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # 计算交叉熵基底（自动处理软标签）
        bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)

        # 计算预测概率p
        p = tf.sigmoid(y_pred)

        # 计算pt：软标签下的正确概率
        pt = y_true * p + (1 - y_true) * (1 - p)
        # pt = tf.exp(-tf.abs(p - y_true))  # 差异越大，pt越小(乘零困境）
        focal_factor = (1 - pt)  ** self.gamma

        # 类别平衡因子（假设0.7视为部分正类，应用alpha）
        # alpha_factor = tf.where(
        #     y_true >= 0.5,  # 假设>=0.5视为正类，根据实际情况调整阈值
        #     self.alpha,
        #     1 - self.alpha
        # )

        alpha_factor = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)

        # 组合损失
        loss = alpha_factor * focal_factor * bce
        return tf.reduce_mean(loss)


# class FocalLoss(tf.keras.losses.Loss):
#     """支持 Focal 机制的损失函数（不带样本权重）"""
#
#     def __init__(self, gamma=2.0, alpha=0.25):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#
#     def call(self, y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)        # 强制转换为 float32 类型
#         # print(f'y_lower1 and upper 0: {tf.less_equal(y_true, 1.0)}')
#         # 使用标准交叉熵基底
#         bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)  # 修改点 1/2
#         # tf.print("y_pred:", y_pred, output_stream=sys.stdout)
#
#         # Focal 调制因子
#         p = tf.sigmoid(y_pred)
#         # tf.print("p:", p, output_stream=sys.stdout)
#         pt = tf.where(tf.logical_or(tf.equal(y_true, 1.0), tf.equal(y_true, 0.7)), p, 1 - p)
#         focal_factor = (1 - pt)  **  self.gamma
#
#         # 类别平衡因子
#         alpha_factor = tf.where(
#             tf.logical_or(tf.equal(y_true, 1.0), tf.equal(y_true, 0.7)), self.alpha, 1 - self.alpha
#         )
#
#         # # 打印 alpha_factor, focal_factor, bce
#         # print_op = tf.print(
#         #     "alpha_factor:", alpha_factor,
#         #     "focal_factor:", focal_factor,
#         #     "bce:", bce,
#         #     output_stream=sys.stdout
#         # )
#         #
#         # # 确保打印操作在损失计算之前执行
#         # with tf.control_dependencies([print_op]):
#         #     loss = tf.reduce_mean(alpha_factor * focal_factor * bce)
#         # return loss
#         return tf.reduce_mean(alpha_factor * focal_factor * bce)




# 设置早停准确率和改善限度
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self,
                 patience=2,
                 train_accuracy_threshold=0.85,
                 best_weights_path='D:/PycharmProjects/wark_by_voice/temp_weights/best_weights.h5',
                 best_model_path='D:/PycharmProjects/wark_by_voice/saved'):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.train_accuracy_threshold = train_accuracy_threshold
        self.best_weights_path = best_weights_path      # 权重保存路径
        # self.best_model_path = best_model_path          # 完整模型保存路径
        self.best = None
        self.wait = 0

        # input_shape = [None, 76, 13, 1]

        # 将 Keras 模型的推理逻辑包装为 TensorFlow 函数
        # @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
        # def serving_default(input_tensor):
        #     return self.model(input_tensor)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = float('inf')  # 假设监控的是损失，如果是准确率则初始化为 -inf

    def on_epoch_end(self, epoch, logs=None):
        # 获取验证集损失和训练集准确率
        val_loss = logs.get('val_loss', float('inf'))   # 如果没有 val_loss，则使用一个很大的值
        train_accuracy = logs.get('rec', 0.0)      # 或者是 'acc'，取决于模型定义

        # 检查训练集准确率是否达到阈值
        if train_accuracy < self.train_accuracy_threshold:
            print(f"\t训练集召回率未达到 {self.train_accuracy_threshold * 100}%，继续训练...")
            return

        # 检查验证集损失是否改善
        if val_loss < self.best:
            # print(f"\t验证损失改善 ({self.best:.4f} → {val_loss:.4f})，保存最佳模型和权重")
            self.best = val_loss
            self.wait = 0
            # 保存权重
            self.model.save_weights(self.best_weights_path)
            # 保存完整模型（包含结构和优化器状态）
            # self.model.save(self.best_model_path)


            # # 保存模型为 SavedModel 格式并指定签名
            # tf.saved_model.save(
            #     self.model,
            #     self.best_model_path,
            #     signatures={"serving_default": serving_default}
            # )
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\t验证集损失在连续 {self.patience} 个轮次内没有改善，训练提前停止。")
                # print(f"\t正在加载最佳模型：{self.best_model_path}")
                # 训练终止时自动加载最佳模型
                # self.model = tf.keras.models.load_model(self.best_model_path)
                self.model.load_weights(self.best_weights_path)





#
# # 设置早停准确率和改善限度
# class CustomEarlyStopping(tf.keras.callbacks.Callback):
#     def __init__(self, patience=2, train_accuracy_threshold=0.8):
#         super(CustomEarlyStopping, self).__init__()
#         self.patience = patience
#         self.train_accuracy_threshold = train_accuracy_threshold
#         self.best_weights = None
#         self.best_weights_path = 'D:/PycharmProjects/wark_by_voice/temp_weights/best_weights.h5'
#         self.best = None
#         self.wait = 0
#
#     def on_train_begin(self, logs=None):
#         self.wait = 0
#         self.best = float('inf')  # 假设监控的是损失，如果是准确率则初始化为 -inf
#
#     def on_epoch_end(self, epoch, logs=None):
#         # 获取验证集损失和训练集准确率
#         val_loss = logs.get('val_loss', float('inf'))   # 如果没有 val_loss，则使用一个很大的值
#         train_accuracy = logs.get('accuracy', 0.0)  # 或者是 'acc'，取决于你的模型定义
#
#         # 检查训练集准确率是否达到阈值
#         if train_accuracy < self.train_accuracy_threshold:
#             print(f"\t训练集准确率未达到 {self.train_accuracy_threshold * 100}%，继续训练...")
#             return
#
#         # 检查验证集损失是否改善
#         if val_loss < self.best:
#             self.best = val_loss
#             self.wait = 0
#             self.model.save_weights(self.best_weights_path)  # 保存权重到磁盘
#             # self.best_weights = self.model.get_weights()
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 self.model.stop_training = True
#                 print(f"\t验证集损失在连续 {self.patience} 个轮次内没有改善，训练提前停止。")
#                 # self.model.set_weights(self.best_weights)  # 恢复最佳权重
#                 self.model.load_weights(self.best_weights_path)  # 从磁盘加载权重