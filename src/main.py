from pathlib import Path

from src.preprocessing.data_preprocessing import normalize_data, preprocess_data
from src.model.model_builder import CustomModel
from src.model.model_trainer import compile_model, train_model
from src.model.utils import CustomEarlyStopping
import seaborn as sns

from tensorflow.keras import layers

from pydub import AudioSegment

import gc
from draw import *
from src.model.export_model import *
from src.preprocessing.Pretreatment import *
from data_loader import *

# 设置 absl 日志级别为 WARNING
# logging.set_verbosity(logging.WARNING)

gc.collect()    # 清理不必要的内存

# 动态分配内存
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12 * 1024)])
                print(f'Using GPU: {gpu}')
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

# 转换为16bit的音频
def convert_to_16bit_wav(input_path, output_path):
    AudioSegment.converter = r"D:\AppData\ffmpeg-2025-01-20-git-504df09c34-essentials_build\bin\ffmpeg.exe"
    audio = AudioSegment.from_wav(input_path)
    audio = audio.set_sample_width(2)  # 转换为 16 位深度
    # audio = tf.audio.decode_wav(audio, desired_channels=1)      # 强制转换为单通道 太卡了这里阻止程序运行
    audio.export(output_path, format="wav")


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

Batch = 10     # 训练样本数量



if __name__ == "__main__":
    print("Eager Execution Enabled:", tf.executing_eagerly())
    configure_gpu()     # 启用gpu,动态分配内存
    # tf.profiler.experimental.start('log_dir')


    # data_dir = '../AISHELL-WakeUp-1-sample/SPEECHDATA/speech/wav'
    data_dir = "D:/PycharmProjects/wark_by_voice/sample_train"

    # 跟踪张量形状变化
    # tf.debugging.set_log_device_placement(True)

    file_paths, labels = load_dataset(data_dir)  # 加载模型训练文件
    dataset, label = preprocess_dataset(file_paths, labels)   # 加載自定義預處理
    label_names = ["1_wake" if x == 1 else "0_non_wake" for x in label]

    # # 迭代一次数据集，确保数据被加载
    # for batch in dataset.take(1):  # 只迭代一个批次
    #     print("数据加载完成，第一个批次：", batch)


    # 设定一个固定的 buffer_size
    buffer_size = 128   # 例如，设定为 128

    # 打乱数据集
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)  # 设置随机种子以保证可复现性

    # 划分为训练集和验证集
    train_size = int(len(file_paths) * 0.8)
    train_ds = dataset.take(train_size).batch(Batch).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(Batch).prefetch(tf.data.AUTOTUNE)

    # 检查数据集的输出形状
    print(f'train_da_shape:{train_ds.element_spec}')

    # 打印数据集的批次形状
    audio_shape = train_ds.element_spec[0].shape
    label_shape = train_ds.element_spec[1].shape
    print(f"Audio shape: {audio_shape}")
    print(f"Label shape: {label_shape}")

    # 数据归一化
    norm_layer = normalize_data(train_ds, val_ds)
    train_ds, val_ds = preprocess_data(train_ds, val_ds, norm_layer)

    print(f"norm Audio shape: {train_ds.element_spec[0].shape}")
    print(f"norm Label shape: {train_ds.element_spec[1].shape}")

    # 扩展维度到四维便于卷积输出
    # 定义一个函数来扩展维度
    def expand_dims(data, label):
        data = tf.expand_dims(data, axis=-1)  # 扩展数据的维度
        return data, label

    # 使用 map 函数将维度扩展应用于每个元素
    train_ds_four = train_ds.map(expand_dims, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_four = val_ds.map(expand_dims, num_parallel_calls=tf.data.AUTOTUNE)

    print("expand train Audio element spec:", train_ds_four.element_spec)
    print("expand Validation dataset element spec:", val_ds_four.element_spec)

    print("卷积输入维度扩展完成")

    # # 强制加载所有数据
    # all_data = list(dataset)  # 将所有数据加载到内存
    # print("所有数据加载完成，数据总数：", len(all_data))

    # 模型构建
    l2_reg = tf.keras.regularizers.L2(l2=0.035)
    # 此处根据实际情况调整 ！！！
    model = CustomModel((None, 26, 13, 1), 2, l2_reg)  # 输入形状应该是 (None, 26, 13, 1)

    # 模型编译（非对称交叉熵，使模型更关注正类
    weights = [1.05, 1.0]
    compile_model(model, weights)

    # 训练模型
    epochs = 2
    callbacks = [
        CustomEarlyStopping(patience=2, train_accuracy_threshold=0.85),
        tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1, update_freq='epoch')
    ]
    history = train_model(model, train_ds_four, val_ds_four, epochs, callbacks)


    export = ExportModel(model)
    tf.saved_model.save(export, "D:/PycharmProjects/wark_by_voice/saved")
    print("end")



    # try:
    #     audio_file_path = str(Path(data_dir)/ '1_wake_words/c_ya_slow_2_10_3_quiet.wav')
    #     input_audio = tf.constant(audio_file_path, dtype=tf.string)
    #     export(input_audio)
    # except Exception as e:
    #     print(f"An error occurred while exporting the model: {e}")
    # finally:
    #     tf.saved_model.save(export, "D:/PycharmProjects/wark_by_voice/saved")
    #     imported = tf.saved_model.load("D:/PycharmProjects/wark_by_voice/saved")
    #     # imported(waveform[tf.newaxis, :])
    #     print("end")




    # 取出频谱数据(一个批次必须大于9)
    # (25, 13)
    for e_g_spectrograms, example_spect_labels in train_ds_four.take(1):
        # example_audio.shape: (10, 16000)
        print(f"example_spectrograms.shape: {e_g_spectrograms.shape}")
        print(f"example_spect_labels.shape: {example_spect_labels.shape}")
        # 绘制前九张的频谱图
        plot_spectrograms(e_g_spectrograms, example_spect_labels, label_names, rows=3, cols=3, figsize=(16, 9))

        input_shape = e_g_spectrograms.shape[1:]
        print('Input shape:', input_shape)
        num_labels = len(label_names)
        break


    def plot_training_history(history, figsize=(16, 6)):
        import matplotlib.pyplot as plt

        # 获取训练和验证损失
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        # 确保长度一致
        min_length = min(len(train_loss), len(val_loss))
        train_loss = train_loss[:min_length]
        val_loss = val_loss[:min_length]

        # 定义 epoch 范围
        epochs = range(1, min_length + 1)

        # 绘图
        plt.figure(figsize=figsize)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # 绘制损失与准确率曲线
    # history 属性是一个字典，记录了训练过程中的各种指标，如损失和准确率
    plot_training_history(history, figsize=(16, 6))


    all_labels_class = ['0_non_wake', '1_wake']
    # 绘制混淆矩阵(可以修改为应用test)
    model.evaluate(val_ds_four, return_dict=True)
    y_pred = model.predict(val_ds_four)
    y_pred = tf.cast(y_pred >= 0.5, tf.int32).numpy().flatten()
    # 真实标签
    y_true = tf.concat(list(val_ds_four.map(lambda s,lab: lab)), axis=0)
    print("True labels:", y_true)
    print("Predicted labels:", y_pred)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=all_labels_class,
                yticklabels=all_labels_class,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    # 结束性能分析
    # tf.profiler.experimental.stop()




# # 分割训练集和验证集(随机打乱方案)
# train_size = int(len(file_paths) * 0.8)
# train_ds = dataset.take(train_size)
# val_ds = dataset.skip(train_size)
#
# # 应用 shuffle 和 cache
# train_ds = train_ds.shuffle(buffer_size=4096).cache().batch(Batch).prefetch(tf.data.AUTOTUNE)
# val_ds = val_ds.cache().batch(Batch).prefetch(tf.data.AUTOTUNE)




# # 小验证
# x = 'D:\\PycharmProjects\\wark_by_voice\\verify\\1_wake\\c_ya_fast_2_10_1_quiet.wav'
#
# # 将输入转换为16bit的音频
# convert_to_16bit_wav(x, x)
#
# x = tf.io.read_file(str(x))
# x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
# x = tf.squeeze(x, axis=-1)
# waveform = x
# # x = get_spectrogram(x)      # 利用对数频谱图
# x = get_mfcc(x)     # 利用mfcc特征图
# x = x[tf.newaxis,...]   # 转化为批次的形式
#
# prediction = model(x)
# print("prediction:", prediction)
#
# x_labels = ['wake_words_probability']
# wake_word_probability = prediction.numpy()[0][0]  # 提取概率值
# plt.bar(x_labels, [wake_word_probability])
#
# plt.title('miya')
# plt.ylabel('Probability')
# plt.show()
#
# display.display(display.Audio(waveform, rate=16000))



# # 创建数据增强管道
# data_augmentation = tf.keras.Sequential([
#     layers.RandomRotation(0.2),  # 随机旋转
#     layers.RandomZoom(0.2),  # 随机缩放
#     layers.RandomTranslation(0.1, 0.1),  # 随机平移
#     layers.RandomContrast(0.2)  # 随机对比度调整
# ])
# # 应用数据增强
# train_spectrogram_ds = train_spectrogram_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y),
#     num_parallel_calls=tf.data.AUTOTUNE
# )

