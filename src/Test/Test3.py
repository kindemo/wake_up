import unittest
import tensorflow as tf
import numpy as np

class TestSTFTTimeSteps(unittest.TestCase):
    def setUp(self):
        # 设置测试参数
        self.frame_length = 400
        self.frame_step = 160

    def test_one_second_audio(self):
        signal_length = 16000
        expected_time_steps = 76
        waveform = np.random.randn(11, 400)  # 单通道输入，形状为 (帧时间步, 帧内采样点)
        # 创建一个随机信号
        single_channel = tf.reshape(waveform, [-1])  # 形状为 (帧时间步 * 帧内采样点,)
        # singal_waveform = waveform.flatten()
        # 计算 STFT
        stft_result = tf.signal.stft(
            single_channel,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length,
            window_fn=tf.signal.hann_window,
            pad_end=False
        )
        # 获取实际的 time_steps
        actual_time_steps = stft_result.shape[0]
        # 断言实际值与预期值是否一致
        self.assertEqual(actual_time_steps, expected_time_steps,
                         f"Failed for signal length {signal_length}. "
                         f"Expected {expected_time_steps}, got {actual_time_steps}.")


if __name__ == '__main__':
    unittest.main()