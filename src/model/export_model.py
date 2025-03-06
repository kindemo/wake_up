import tensorflow as tf

from src.preprocessing.Pretreatment import split_audio_channels
from src.preprocessing.wave_processing import squeeze as squeezing


class ExportModel(tf.Module):
    def __init__(self, model, frame_length=400, n_mfcc=13, num_win=31):
        super().__init__()
        self.model = model
        self.frame_length = frame_length
        self.n_mfcc = n_mfcc
        self.num_win = num_win

        # 注册方法的签名
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        # 确保 x 是一个 TensorFlow 张量
        if isinstance(x, str):
            x = tf.constant(x, dtype=tf.string)

        # 如果输入是字符串（文件路径），则加载并解码音频
        s_rate = 16000
        if x.dtype == tf.string:
            try:
                # 用tensorflow自带的库
                audio_binary = tf.io.read_file(x)
                wave, s_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
                # 将音频数据转换为 tf.float32 类型
                wave = tf.cast(wave, dtype=tf.float32)
                # print("ExportModel’s wave shape before squeeze:", wave.shape)
                wave = squeezing(wave)
            except Exception as e:
                print(f"Error loading audio file: {e}")
                wave = tf.zeros(16000, dtype=tf.float32)
        elif x.dtype == tf.float32:
            # 需要改进
            wave = x
            wave = tf.cast(wave, dtype=tf.float32)
        else:
            raise ValueError("Unsupported input type. Expected file path or audio data.")

        s_rate = tf.cast(s_rate, dtype=tf.float32)

        # 将音频划分为多个通道
        channels = split_audio_channels(wave, s_rate, self.frame_length, self.n_mfcc, self.num_win)


        # # 使用 tf.TensorArray 替代 Python 列表
        # results = tf.TensorArray(dtype=tf.float32, size=tf.shape(channels)[0])

        # # 遍历每个通道
        # for i in tf.range(tf.shape(channels)[0]):
        #     channel = channels[i]
        #     # print("Channel", i, "shape:", channels[i].shape)
        #     # print("Channel", i, "data:", channels[i][:50])  # 查看前 50 个特征
        #     channel = tf.expand_dims(channel, axis=0)  # 添加批次维度
        #     channel = tf.expand_dims(channel, axis=-1)  # 添加通道维度
        #     result = self.model(channel, training=False)  # 模型预测
        #     result = tf.squeeze(result, axis=0)  # 去掉批次维度
        #     results = results.write(i, result)  # 将结果写入 TensorArray


        channels = tf.expand_dims(channels, axis=-1)  # 添加通道维度
        results = self.model(channels, training=False)  # 模型预测
        # print(results.shape)

        # 需要交换维度和通道位置（关键步骤！）
        # channels = tf.transpose(channels, [0, 2, 1, 3])  # 输出形状 (num_channels, 13, 76, 1)
        # 将 TensorArray 转换为张量
        # results = results.stack()  # 形状为 (num_channels, num_classes)

        # 设置阈值并判断类别
        threshold = 0.5
        class_ids = tf.cast(results >= threshold, dtype=tf.int32)  # 形状为 (num_channels, num_classes)

        return {'predictions': results, 'class_ids': class_ids}









# try:
#     # 将音频数据转换为 tf.float32 类型
#     s_rate = tf.convert_to_tensor(s_rate, dtype=tf.float32)
#     wave = tf.convert_to_tensor(wave, dtype=tf.float32)
#
#     # 调用降噪和端点检测函数
#     waveform = del_signal_ini(s_rate, wave)  # 降噪端点检测
# except Exception as e:
#     print(f"Error deleting signal: {e}")
#     waveform = tf.zeros(16000, dtype=tf.float32)
#
# # 分帧加窗
# try:
#     windows = get_windows(waveform, num_windows=self.num_win)  # 小帧无重叠，大帧重叠分帧
# except Exception as e:
#     print(f"Error getting windows: {e}")
#     windows = tf.zeros((self.num_win, self.frame_length, 1), dtype=tf.float32)
#
# # 提取特征获取帧的多通道，假设返回形状为(num_channels, num_windows, n_mfcc)
# try:
#     channels = get_mfcc(windows, num_windows=self.num_win)
# except Exception as e:
#     print(f"Error getting MFCC features: {e}")
#     sum_mfcc_frames = int(tf.math.floor((self.frame_length * self.num_win - self.frame_length) / 160)) + 1
#     channels = tf.zeros((1, sum_mfcc_frames, self.n_mfcc), dtype=tf.float32)
