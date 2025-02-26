import tensorflow as tf

# 加权二元交叉熵
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = -weights[0] * y_true * tf.math.log(y_pred + 1e-7) - weights[1] * (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
        return tf.reduce_mean(loss)
    return loss


# 设置早停准确率和改善限度
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=2, train_accuracy_threshold=0.8):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.train_accuracy_threshold = train_accuracy_threshold
        self.best_weights = None
        self.best_weights_path = 'D:/PycharmProjects/wark_by_voice/temp_weights/best_weights.h5'
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
