import numpy as np
import librosa
import matplotlib.pyplot as plt

import numpy as np

def get_windows(waveform, frame_length=400, frame_step=160, num_windows=10):
    """
    将音频信号分帧并按滑动窗口拼接为窗。
    如果最后一个窗不足指定数量的帧，则填充为0。

    参数:
        waveform: 输入的音频信号（1D numpy数组）。
        frame_length: 每个帧的样本数。
        frame_step: 每个帧的步长。
        num_windows: 每个拼接窗的帧数。

    返回:
        输出的shape为（帧长 400, 窗大小 10, 窗数量 None）。
    """
    # 分帧
    frames = []
    start = 0
    while start + frame_length <= len(waveform):
        frame = waveform[start:start + frame_length]
        frames.append(frame)
        start += frame_step

    # 如果最后一个帧不足 frame_length，填充为0
    if start < len(waveform):
        remaining = waveform[start:]
        padded_remaining = np.pad(remaining, (0, frame_length - len(remaining)), mode='constant')
        frames.append(padded_remaining)

    # 将帧按滑动窗口拼接为窗
    windows = []
    for i in range(len(frames) - num_windows + 1):
        window = frames[i:i + num_windows]
        windows.append(np.stack(window, axis=0))  # 将窗内的帧堆叠起来

    # 如果最后一个窗不足 num_windows 个帧，填充为0
    if len(frames) % num_windows != 0:
        last_window = frames[-num_windows:]
        padded_last_window = np.pad(last_window, ((0, num_windows - len(last_window)), (0, 0)), mode='constant')
        windows.append(padded_last_window)

    # 转换为 NumPy 数组并调整形状
    concatenated_windows = np.stack(windows, axis=-1)  # (帧长, 窗大小, 窗数量)
    return concatenated_windows


def enframe(signal, frame_len=400, frame_shift=160, win_func=np.hamming):
    """
    分帧并加窗（末尾不足一帧直接丢弃）
    :param signal: 输入音频信号 (wave)
    :param frame_len: 帧长（采样点数）
    :param frame_shift: 帧移（采样点数）
    :param win_func: 窗函数，默认为汉明窗
    :return: 分帧并加窗后的信号 (num_frames, 400)
    """
    num_samples = len(signal)
    if num_samples < frame_len:
        return np.zeros((0, frame_len))  # 直接返回空数组
    num_frames = (num_samples - frame_len) // frame_shift + 1  # 严格丢弃末尾不足的帧
    frames = np.zeros((num_frames, frame_len))

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len
        frames[i, :] = signal[start:end] * win_func(frame_len)  # 直接截取

    return frames


# 示例：加载音频
audio_path = '../../verify/0_non_wake/0001_M02_01_fast_0009.wav'
y, sr = librosa.load(audio_path, sr=16000)

# 设置参数
frame_len = int(0.025 * sr)  # 25ms
frame_shift = int(0.010 * sr)  # 10ms

# 分帧并加窗
frames = enframe(y, frame_len, frame_shift)
print("Frames shape:", frames.shape)  # 输出分帧后的形状



# 计算频谱图（基于分帧后的数据）
n_fft = 512
mag_frames = np.abs(np.fft.rfft(frames, n_fft))  # 幅度谱
pow_frames = (1.0 / n_fft) * (mag_frames ** 2)   # 功率谱
log_pow_frames = 20 * np.log10(pow_frames + 1e-10)  # 对数功率谱

# 可视化频谱图（修正标题和坐标轴）
plt.figure(figsize=(12, 6))
plt.imshow(log_pow_frames.T, cmap='jet', aspect='auto', origin='lower')
plt.title('Spectrogram (Per Frame)')  # 修正标题
plt.xlabel('Frame Index')
plt.ylabel('Frequency Bin')
plt.colorbar(label='Power/Frequency (dB)')
plt.show()

# 拼接连续五帧（确保不足五帧时丢弃末尾）
num_frames_to_concatenate = 5
concatenated_frames = []

for i in range(frames.shape[0] - num_frames_to_concatenate + 1):
    concatenated_frame = frames[i:i+5].reshape(-1)
    concatenated_frames.append(concatenated_frame)

concatenated_frames = np.array(concatenated_frames)
print("Concatenated frames shape:", concatenated_frames.shape)


# 将拼接后的频谱图转换为对数尺度以便更好地可视化
log_pow_frames = 20 * np.log10(pow_frames + 1e-10)  # 避免除零


# 可视化频谱图
plt.figure(figsize=(12, 6))
plt.imshow(log_pow_frames.T, cmap='jet', aspect='auto')
plt.title('Concatenated Spectrogram')
plt.xlabel('Frame Index')
plt.ylabel('Frequency Bin')
plt.colorbar(label='Power/Frequency (dB)')
plt.show()