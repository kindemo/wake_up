import tensorflow as tf
import numpy as np
import librosa
from pathlib import Path


class AudioAugmenter:
    def __init__(self, noise_paths, target_sr=16000):
        self.target_sr = target_sr
        self.noise_data = self._load_and_preprocess_noise(noise_paths)

        # 修复滤波器初始化问题
        self._initialize_filters()

        # 增强概率参数
        self.augment_probs = {
            'speed': 0.8,
            'shift': 0.5,
            'crop': 0.8,
            'noise': 0.8,
            'gain': 0.5,
            'filter': 0.3
        }

    def _load_and_preprocess_noise(self, noise_dir):
        """修复路径类型问题"""
        noise_dir = Path(noise_dir)  # 确保转换为Path对象
        if not noise_dir.is_dir():
            raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

        wav_paths = list(noise_dir.rglob("*.wav"))
        if not wav_paths:
            raise ValueError(f"No WAV files in: {noise_dir}")

        # 转换为字符串路径列表
        str_paths = [str(p) for p in wav_paths]

        noise_arrays = []
        max_length = 10 * self.target_sr
        for p in str_paths:  # 使用字符串路径
            # 使用TF文件读取
            raw_audio = tf.io.read_file(p)
            audio, _ = tf.audio.decode_wav(raw_audio)
            audio = tf.squeeze(audio, axis=-1)

            # 填充/截断处理
            current_length = tf.shape(audio)[0]
            padding = [[0, tf.maximum(0, max_length - current_length)]]
            padded_audio = tf.pad(audio, padding)[:max_length]

            noise_arrays.append(padded_audio)

        return tf.concat(noise_arrays, axis=0)

    def _initialize_filters(self):
        """修复卷积滤波器初始化"""
        self.filters = []
        filter_length = 2048
        for _ in range(10):
            # 生成合规的卷积核形状 [filter_length, in_channels, out_channels]
            kernel = tf.random.normal([filter_length, 1, 1],
                                      stddev=0.1) * tf.signal.hann_window(filter_length)[:, tf.newaxis, tf.newaxis]
            self.filters.append(kernel)

    def _apply_random_filter(self, audio):
        """修复卷积实现"""
        # 添加批次和通道维度 [batch, length, channels]
        audio_4d = audio[tf.newaxis, :, tf.newaxis]
        conv = tf.nn.conv1d(
            audio_4d,
            filters=self.filters[np.random.randint(len(self.filters))],
            stride=1,
            padding="SAME"
        )
        return tf.squeeze(conv, axis=[0, -1])

    def _speed_perturbation(self, audio):
        """完全兼容的相位声码器实现"""
        rate = tf.random.uniform([], 0.9, 1.1)
        original_length = tf.shape(audio)[0]

        # STFT参数
        frame_length = 256
        frame_step = 64
        fft_length = 256

        # 执行STFT
        stft = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )

        # 分解幅度和相位
        magnitude = tf.abs(stft)
        phase = tf.math.angle(stft)

        # 相位差分计算
        phase_diff = phase[:, 1:] - phase[:, :-1]
        phase_diff = tf.where(phase_diff < 0, phase_diff + 2 * np.pi, phase_diff)

        # 构建调整后的相位
        adjusted_phase = tf.cumsum(phase_diff * rate, axis=1)

        # 创建复数相位因子（显式类型转换）
        complex_phase = tf.complex(
            tf.cos(adjusted_phase),
            tf.sin(adjusted_phase)
        )

        # 显式转换幅度为复数类型
        complex_magnitude = tf.cast(magnitude[:, 1:-1], tf.complex64)

        # 构建新的STFT
        new_stft = tf.concat([
            tf.complex(magnitude[:, 0:1], 0.0),  # 首帧
            complex_magnitude * complex_phase,  # 中间帧
            tf.complex(magnitude[:, -1:], 0.0)  # 末帧
        ], axis=1)

        # 逆STFT
        stretched = tf.signal.inverse_stft(
            new_stft,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )

        return self._match_length(stretched, original_length)

    def _time_shift(self, audio):
        """改进的时移处理（循环填充）"""
        max_shift = int(0.2 * self.target_sr)
        shift = tf.random.uniform(shape=(), minval=-max_shift, maxval=max_shift, dtype=tf.int32)
        return tf.roll(audio, shift=shift, axis=0)


    def _time_shift(self, audio):
        """改进的时移处理（循环填充）"""
        max_shift = int(0.2 * self.target_sr)
        shift = tf.random.uniform(shape=(), minval=-max_shift, maxval=max_shift, dtype=tf.int32)
        return tf.roll(audio, shift=shift, axis=0)

    def _dynamic_crop(self, audio):
        """带平滑过渡的动态裁剪"""
        original_length = tf.shape(audio)[0]
        crop_length = tf.random.uniform(
            shape=(),
            minval=int(0.8 * tf.cast(original_length, tf.float32)),
            maxval=original_length,
            dtype=tf.int32
        )

        # 添加10ms的淡入淡出
        fade_length = int(0.01 * self.target_sr)
        fade_in = tf.linspace(0.0, 1.0, fade_length)
        fade_out = tf.linspace(1.0, 0.0, fade_length)

        start = tf.random.uniform(shape=(), maxval=original_length - crop_length, dtype=tf.int32)
        cropped = audio[start:start + crop_length]

        # 应用淡入淡出
        cropped = tf.concat([
            cropped[:fade_length] * fade_in,
            cropped[fade_length:-fade_length],
            cropped[-fade_length:] * fade_out
        ], axis=0)

        return tf.pad(cropped, [[0, original_length - crop_length]])


    def _match_length(self, audio, target_length):
        """改进长度匹配逻辑"""
        current_length = tf.shape(audio)[0]
        pad_length = target_length - current_length

        return tf.cond(
            pad_length > 0,
            lambda: tf.pad(audio, [[0, pad_length]]),
            lambda: audio[:target_length]
        )

    def _add_noise(self, audio):
        """改进的噪声添加（带动态范围控制）"""
        audio_length = tf.shape(audio)[0]
        noise_length = tf.shape(self.noise_data)[0]

        # 处理噪声长度不足的情况
        start_idx = tf.cond(
            noise_length > audio_length,
            lambda: tf.random.uniform([], 0, noise_length - audio_length, dtype=tf.int32),
            lambda: 0
        )

        noise_segment = tf.cond(
            noise_length > audio_length,
            lambda: self.noise_data[start_idx:start_idx + audio_length],
            lambda: tf.tile(self.noise_data, [audio_length // noise_length + 1])[:audio_length]
        )

        # 动态范围匹配
        audio_rms = tf.sqrt(tf.reduce_mean(audio ** 2))
        noise_rms = tf.sqrt(tf.reduce_mean(noise_segment ** 2))
        scaled_noise = noise_segment * (audio_rms / (noise_rms + 1e-7)) * tf.random.uniform([], 0.05, 0.2)

        return tf.clip_by_value(audio + scaled_noise, -1.0, 1.0)

    def _random_gain(self, audio):
        """带自动增益控制的随机增益"""
        gain = tf.random.uniform([], 0.8, 1.2)
        return tf.clip_by_value(audio * gain, -1.0, 1.0)

    def augment(self, audio):
        """添加输入验证和类型强制转换"""
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        audio = tf.ensure_shape(audio, [None])

        # 使用tf.cond实现概率控制
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['speed'],
            lambda: self._speed_perturbation(audio),
            lambda: audio
        )
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['shift'],
            lambda: self._time_shift(audio),
            lambda: audio
        )
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['crop'],
            lambda: self._dynamic_crop(audio),
            lambda: audio
        )
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['noise'],
            lambda: self._add_noise(audio),
            lambda: audio
        )
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['gain'],
            lambda: self._random_gain(audio),
            lambda: audio
        )
        audio = tf.cond(
            tf.random.uniform(()) < self.augment_probs['filter'],
            lambda: self._apply_random_filter(audio),
            lambda: audio
        )
        return tf.clip_by_value(audio, -1.0, 1.0)



if __name__ == "__main__":
    # 使用示例
    # noise_paths = Path('D:/PycharmProjects/wark_by_voice/noisex-92-master')  # 加载NOISEX-92数据集路径
    # # 初始化增强器
    # augmenter = AudioAugmenter(
    #     noise_paths=noise_paths,
    #     target_sr=16000
    # )
    #
    # # 使用示例
    # audio, sr = librosa.load("D:/PycharmProjects/wark_by_voice/miya_long3.wav", sr=16000)
    # augmented = augmenter.augment(tf.convert_to_tensor(audio, tf.float32))
    # # 验证长度保持
    # assert len(augmented) == len(audio)  # True
    # print(f"augmented: {augmented.shape}")


    # 测试用例
    def test_augmenter():
        # 生成测试音频（3秒，16000Hz）
        test_audio = tf.random.uniform([16000 * 3], -0.5, 0.5)

        # 初始化增强器
        augmenter = AudioAugmenter(Path('D:/PycharmProjects/wark_by_voice/noisex-92-master'))

        # 验证输出形状
        augmented = augmenter.augment(test_audio)
        assert augmented.shape == test_audio.shape, "Shape mismatch"

        # 验证数值范围
        assert tf.reduce_max(augmented) <= 1.0, "Clipping failure"
        assert tf.reduce_min(augmented) >= -1.0, "Clipping failure"

        # 验证滤波器应用
        filtered = augmenter._apply_random_filter(test_audio)
        assert filtered.shape == test_audio.shape, "Filter shape error"

        print("All tests passed!")


    # 执行测试
    test_augmenter()