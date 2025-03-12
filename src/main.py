from pathlib import Path

from src.model.export_model import ExportModel
from src.preprocessing.data_preprocessing import normalize_data, preprocess_data
from src.model.model_builder import EnhancedWakeModel
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
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=13 * 1024)])
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


Batch = 128     # 训练样本数量
epochs = 10
f_block = 16     # 每次从一类文件中取出几个
a_balance = 0.7        # 控制样本平衡(更偏爱优化正类)
gamma_punish = 0.5
l2_reg = 1e-4

# # 设定一个固定的 buffer_size
# buffer_size = 5120   # 缓冲区大小设定为 5120



if __name__ == "__main__":
    print("Eager Execution Enabled:", tf.executing_eagerly())
    configure_gpu()     # 启用gpu,动态分配内存
    # tf.profiler.experimental.start('log_dir')

    # data_dir = "D:/PycharmProjects/wark_by_voice/AISHELL-WakeUp-1-sample/SPEECHDATA/speech/wav"
    data_dir = "D:/PycharmProjects/wark_by_voice/sample_train"
    # data_dir = "D:/PycharmProjects/wark_by_voice/verify"


    # 跟踪张量形状变化
    # tf.debugging.set_log_device_placement(True)

    file_paths, labels = load_dataset(data_dir)  # 加载模型训练文件
    dataset = create_interleaved_dataset(file_paths, labels, block_size=f_block)

    # # 打印前几个元素验证标签分配
    # for element in dataset.take(15):
    #     file_path, label = element
    #     print(f"File path: {file_path.numpy().decode('utf-8')}, Label: {label.numpy()}")


    dataset, label = preprocess_dataset(dataset, num_win=31)   # 加載自定義預處理

    # # 迭代一次数据集，确保数据被加载
    # for batch in dataset.take(1):  # 只迭代一个批次
    #     print("数据加载完成，第一个批次：", batch)


    # 划分为训练集和验证集
    # 尝试获取 cardinality
    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.UNKNOWN_CARDINALITY:
        # 手动计算大小
        total_size = sum(1 for _ in dataset)
    elif cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("数据集无限，无法划分")
    else:
        total_size = cardinality
        print(f"total size: {total_size}")

    train_size = int(total_size * 0.8)
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

    # print(f"norm Audio shape: {train_ds.element_spec[0].shape}")
    # print(f"norm Label shape: {train_ds.element_spec[1].shape}")

    # 扩展维度到四维便于卷积输出
    # 定义一个函数来扩展维度
    def expand_dims(data, label):
        data = tf.expand_dims(data, axis=-1)  # 扩展数据的维度
        return data, label

    # 使用 map 函数将维度扩展应用于每个元素
    train_ds_four = train_ds.map(expand_dims, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_four = val_ds.map(expand_dims, num_parallel_calls=tf.data.AUTOTUNE)

    # 打印前几个元素验证标签分配
    for element in train_ds_four.take(1):
        slices, label = element
        print(f"batch slices: {tf.shape(slices).numpy()}, Label: {label.numpy()}")

    print("expand train Audio element spec:", train_ds_four.element_spec)
    print("expand Validation dataset element spec:", val_ds_four.element_spec)

    print("卷积输入维度扩展定义完成")

    # # 强制加载所有数据
    # all_data = list(dataset)  # 将所有数据加载到内存
    # print("所有数据加载完成，数据总数：", len(all_data))

    # 模型构建
    # 此处根据实际情况调整 ！！！
    model = EnhancedWakeModel((None, 31, 400, 1), l2_reg)  # 输入形状应该是 (76, 13, 1)

    # 模型编译（非对称交叉熵，使模型更关注正类
    compile_model(model, gamma_punish, a_balance)

    # 训练模型
    callbacks = [
        CustomEarlyStopping(patience=3, train_accuracy_threshold=0.9),
        tf.keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=1, update_freq='epoch')
    ]
    history = train_model(model, train_ds_four, val_ds_four, epochs, callbacks)


    export = ExportModel(model, num_win=31)
    # tf.saved_model.save(export, "D:/PycharmProjects/wark_by_voice/saved")

    # 保存模型, 显式声明签名
    tf.saved_model.save(
        export,
        "D:/PycharmProjects/wark_by_voice/saved",
        signatures={
            "file_input": export.file_signature,
            "mfcc_input": export.mfcc_signature
        }
    )

    print("end")


    # 取出频谱数据(一个批次必须大于9)
    # (batch, 76, 13) 三维
    for e_g_spectrograms, example_spect_labels in train_ds.take(1):
        # example_audio.shape: (10, 16000)
        # print(f"example_spectrograms.shape: {e_g_spectrograms.shape}")
        # print(f"example_spect_labels.shape: {example_spect_labels.shape}")
        # 绘制前九张的频谱图
        plot_spectrograms(e_g_spectrograms, example_spect_labels, rows=3, cols=3, figsize=(16, 9))
        input_shape = e_g_spectrograms.shape
        print('Input shape:', input_shape)
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

    print("开始输出混淆矩阵：")
    all_labels_class = ['0_non_wake', '1_wake']
    # 绘制混淆矩阵(可以修改为应用test)
    model.evaluate(val_ds_four, return_dict=True)
    y_pred = model.predict(val_ds_four)
    y_pred_class = tf.cast(y_pred >= 0.5, tf.int32).numpy().flatten()

    # 真实标签
    y_true = tf.concat(list(val_ds_four.map(lambda s,lab: lab)), axis=0)
    print("True labels:", y_true)
    print("Predicted value:", y_pred)
    print("Predicted labels:", y_pred_class)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_class)
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