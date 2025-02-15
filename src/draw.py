import matplotlib.pyplot as plt
from Spectrum_processing import plot_spectrogram
import numpy as np

def plot_spectrograms(spectrograms, labels, label_names, rows=3, cols=3, figsize=(16, 9)):
    """
    绘制频谱图的函数。

    参数:
    - spectrograms: 一个 NumPy 数组，包含多个频谱图。
    - labels: 一个 NumPy 数组，包含每个频谱图对应的标签索引。
    - label_names: 一个列表，包含标签的名称。
    - rows: 子图的行数，默认为 3。
    - cols: 子图的列数，默认为 3。
    - figsize: 图的大小，默认为 (16, 9)。
    """
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(n):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        # 绘制频谱图
        plot_spectrogram(spectrograms[i].numpy(), ax)
        # print(f"Spectrogram {i} shape: {spectrograms.shape}")

        # 设置标题
        label_index = labels[i].numpy()
        label_name = label_names[label_index]
        ax.set_title(label_name)

    plt.tight_layout()
    plt.show()


# 绘制取出的前九个音频波形
def plot_audio_waveforms(audio_signals, labels, label_names, rows=3, cols=3, figsize=(16, 10)):
    n = rows * cols
    plt.figure(figsize=figsize)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        audio_signal = audio_signals[i]
        plt.plot(audio_signal)  # 绘制音频信号的波形图
        label_index = labels[i]
        label_name = label_names[label_index]
        plt.title(label_name)
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])

    plt.tight_layout()
    plt.show()


def plot_waveform_and_spectrogram(waveform, spectrogram, label, figsize=(12, 8)):
    """
    绘制音频波形图和频谱图的函数。

    参数:
    - waveform: 一个 NumPy 数组，包含音频信号。
    - spectrogram: 一个 NumPy 数组，包含音频的频谱图。
    - label: 标签名称。
    - figsize: 图的大小，默认为 (12, 8)。
    """
    fig, axes = plt.subplots(2, figsize=figsize)

    # 第一个子图：绘制音频波形图
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, waveform.shape[0]])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')

    # 第二个子图：绘制频谱图
    plot_spectrogram(spectrogram, axes[1])
    axes[1].set_title('Spectrogram')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency (log scale)')

    # 设置总标题
    plt.suptitle(label.title())
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应总标题
    plt.show()


def plot_training_history(history, figsize=(16, 6)):
    """
    绘制训练过程中的损失和准确率曲线。

    参数:
    - history: 一个 `tf.keras.callbacks.History` 对象，包含训练过程中的指标。
    - figsize: 图的大小，默认为 (16, 6)。
    """
    metrics = history.history
    epochs = history.epoch

    plt.figure(figsize=figsize)

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['loss'], label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')
    plt.title('Loss Curves')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, 100 * np.array(metrics['accuracy']), label='Training Accuracy')
    plt.plot(epochs, 100 * np.array(metrics['val_accuracy']), label='Validation Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.show()