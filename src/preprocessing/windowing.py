import tensorflow as tf


def get_windows(waveform: tf.Tensor,
                frame_length = 400,
                frame_step = 400,
                num_windows = 11) -> tf.Tensor:
    """
    TensorFlow版音频无重叠分帧+滑动拼接窗（修正类型错误版）

    参数:
        waveform: 输入的音频信号（1D Tensor）
        frame_length: 每个帧的样本数（强制转换为整数）
        frame_step: 分帧步长（强制转换为整数）
        num_windows: 每个拼接窗包含的小帧数（强制转换为整数）

    返回:
        Tensor形状为（num_windows, frame_length, num_windows_available）
    """
    # 强制参数转换为整数
    frame_length = tf.cast(frame_length, tf.int32)
    frame_step = tf.cast(frame_step, tf.int32)
    num_windows = tf.cast(num_windows, tf.int32)

    # 确保输入是一维Tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    waveform = tf.reshape(waveform, [-1])

    # 计算最小所需长度（使用整数运算）
    min_required_length = (num_windows - 1) * frame_step + frame_length
    min_required_length = tf.cast(min_required_length, tf.int32)
    original_length = tf.shape(waveform)[0]

    # 动态检查输入长度
    with tf.control_dependencies([
        tf.debugging.assert_greater_equal(
            original_length,
            min_required_length,
            message=f"Input too short: needs {min_required_length} samples")
    ]):
        waveform = tf.identity(waveform)

    def calculate_frames(wave_len):
        return tf.cast(
            tf.math.ceil(
                (tf.cast(wave_len, tf.float32) - tf.cast(frame_length, tf.float32)) / tf.cast(frame_step, tf.float32)
            ),
            tf.int32
        ) + 1

    num_frames = calculate_frames(original_length)
    total_length = (num_frames - 1) * frame_step + frame_length

    # 修正2：确保填充量为整数类型
    pad_amount = tf.maximum(total_length - original_length, 0)
    waveform_padded = tf.pad(waveform, [[0, pad_amount]], constant_values=0.0)

    # 执行分帧
    frames = tf.signal.frame(waveform_padded, frame_length, frame_step, pad_end=False, axis=0)
    num_frames_actual = tf.shape(frames)[0]

    # 计算滑动窗口参数
    window_stride = num_windows // 2
    num_windows_available = (num_frames_actual - num_windows) // window_stride + 1

    # 检查窗口数量有效性
    with tf.control_dependencies([
        tf.debugging.assert_greater_equal(
            num_windows_available,
            1,
            message="Not enough frames to create windows")
    ]):
        frames = tf.identity(frames)

    start_indices = window_stride * tf.range(num_windows_available, dtype=tf.int32)
    window_indices = start_indices[:, tf.newaxis] + tf.range(num_windows, dtype=tf.int32)[tf.newaxis, :]

    # 收集窗口数据并调整维度
    windows = tf.gather(frames, window_indices)
    return tf.transpose(windows, [1, 2, 0])




class TestGetWindows(tf.test.TestCase):
    def test_normal_case(self):
        """测试正常情况：音频长度足够，参数合理"""
        waveform = tf.random.normal([15000])  # 音频信号长度为 15000
        frame_length = 400
        frame_step = 400
        num_windows = 11

        result = get_windows(waveform, frame_length, frame_step, num_windows)

        # 验证输出形状
        expected_shape = (num_windows, frame_length, 6)  # 音频长度为15000，可以生成6个拼接窗
        self.assertEqual(result.shape, expected_shape)

    def test_boundary_case(self):
        """测试边界情况：音频长度刚好满足要求"""
        waveform = tf.random.normal([4000])  # 音频长度刚好满足要求
        frame_length = 400
        frame_step = 400
        num_windows = 10

        # 验证参数
        min_required_length = (num_windows - 1) * frame_step + frame_length
        min_required_length = tf.cast(min_required_length, tf.int32)
        self.assertEqual(min_required_length, 4000)  # 确保最小所需长度为 4000

        result = get_windows(waveform, frame_length, frame_step, num_windows)

        # 验证输出形状
        expected_shape = (num_windows, frame_length, 1)  # 只能生成1个拼接窗
        self.assertEqual(result.shape, expected_shape)

    def test_insufficient_length(self):
        """测试音频长度不足的情况"""
        waveform = tf.random.normal([3000])  # 音频长度不足
        frame_length = 400
        frame_step = 400
        num_windows = 10

        # 使用 unittest 提供的 assertRaises 方法来捕获错误
        with self.assertRaises(tf.errors.InvalidArgumentError):
            get_windows(waveform, frame_length, frame_step, num_windows)

    def test_tensor_input(self):
        """测试输入为 TensorFlow Tensor 的情况"""
        waveform = tf.random.normal([10000])  # 音频信号长度为 10000
        frame_length = 400
        frame_step = 400
        num_windows = 10

        result = get_windows(waveform, frame_length, frame_step, num_windows)

        # 验证输出形状
        expected_shape = (num_windows, frame_length, 4)  # 音频长度为10000，可以生成4个拼接窗
        self.assertEqual(result.shape, expected_shape)

    def test_extreme_case(self):
        """测试极端情况：frame_length 或 frame_step 非常大"""
        waveform = tf.random.normal([1000])  # 音频信号长度为 1000
        frame_length = 1000
        frame_step = 1000
        num_windows = 10

        with self.assertRaises(tf.errors.InvalidArgumentError):
            get_windows(waveform, frame_length, frame_step, num_windows)


if __name__ == '__main__':
    tf.test.main()
