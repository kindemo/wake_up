import tensorflow as tf

# def get_windows(waveform: tf.Tensor,
#                 frame_length=400,
#                 frame_step=400,
#                 num_windows=31) -> tf.Tensor:
#     """
#     原始版本，舍弃大量帧
#     :param waveform:
#     :param frame_length:
#     :param frame_step:
#     :param num_windows:
#     :return:
#     """
#     # 强制参数转换为整数
#     frame_length = tf.cast(frame_length, tf.int32)
#     frame_step = tf.cast(frame_step, tf.int32)
#     num_windows = tf.cast(num_windows, tf.int32)
#
#     # 确保输入是一维Tensor
#     waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
#     waveform = tf.reshape(waveform, [-1])
#     original_length = tf.shape(waveform)[0]
#
#     # 计算最小所需长度并补零
#     min_required_length = (num_windows - 1) * frame_step + frame_length
#     pad_needed = tf.maximum(min_required_length - original_length, 0)
#     waveform_padded = tf.pad(waveform, [[0, pad_needed]])
#
#     # 分帧（允许末尾补零）
#     frames = tf.signal.frame(waveform_padded, frame_length, frame_step, pad_end=True, axis=0)
#     num_frames_actual = tf.shape(frames)[0]
#
#     # 计算滑动窗口参数
#     window_stride = num_windows // 2
#     num_windows_available = (num_frames_actual - num_windows) // window_stride + 1
#
#     # 确保至少能生成一个窗口
#     tf.debugging.assert_greater_equal(
#         num_windows_available,
#         1,
#         message="Not enough frames to create windows"
#     )
#
#     # 收集窗口数据
#     start_indices = window_stride * tf.range(num_windows_available, dtype=tf.int32)
#     window_indices = start_indices[:, tf.newaxis] + tf.range(num_windows, dtype=tf.int32)[tf.newaxis, :]
#     windows = tf.gather(frames, window_indices)
#
#     return tf.transpose(windows, [1, 2, 0])




def get_windows(waveform: tf.Tensor,
                frame_length=400,
                frame_step=400,
                num_windows=31) -> tf.Tensor:
    """
    改进版音频分帧处理，实现两阶段标准化分帧

    参数:
        waveform: 输入的1D音频信号Tensor
        frame_length: 每个小帧的样本数
        frame_step: 小帧之间的步长
        num_windows: 每个大窗口包含的小帧数

    返回:
        Tensor形状为（num_big_frames, num_windows, frame_length）
    """
    # 参数类型转换
    frame_length = tf.cast(frame_length, tf.int32)
    frame_step = tf.cast(frame_step, tf.int32)
    num_windows = tf.cast(num_windows, tf.int32)

    # 输入预处理
    waveform = tf.reshape(waveform, [-1])
    original_length = tf.shape(waveform)[0]

    # 第一阶段参数计算
    big_frame_length = num_windows * frame_length
    big_step = (num_windows // 2) * frame_step
    threshold = (big_frame_length * 6) // 10  # 60%阈值

    # 短波形处理（直接补零）
    def handle_short():
        pad = big_frame_length - original_length
        padded = tf.pad(waveform, [[0, pad]])
        return tf.expand_dims(padded, 0)

    # 长波形处理（动态分帧）
    def handle_long():
        # 生成候选起始点
        starts = tf.range(0, original_length, big_step)

        # 计算有效长度并过滤
        ends = starts + big_frame_length
        valid_lengths = tf.where(
            ends <= original_length,
            big_frame_length,
            original_length - starts
        )
        keep_mask = valid_lengths >= threshold
        kept_starts = tf.boolean_mask(starts, keep_mask)

        # 提取并补零窗口
        def get_window(start):
            window = waveform[start:start + big_frame_length]
            return tf.pad(window, [[0, big_frame_length - tf.shape(window)[0]]])

        return tf.map_fn(get_window, kept_starts, fn_output_signature=tf.float32)

    # 执行分阶段处理
    big_frames = tf.cond(
        original_length < big_frame_length,
        handle_short,
        handle_long
    )

    # 第二阶段：分小帧并调整维度
    small_frames = tf.signal.frame(
        big_frames,
        frame_length=frame_length,
        frame_step=frame_step,

        pad_end=False,
        axis=1
    )
    # 应用窗函数
    window = tf.signal.hamming_window(frame_length, dtype=tf.float32)  # 使用汉明窗
    small_frames = small_frames * window  # 将窗函数应用于每个小帧

    # return tf.transpose(small_frames, [1, 2, 0])
    return small_frames




class TestGetWindows(tf.test.TestCase):
    def test_normal_case(self):
        """测试正常情况：音频长度足够，参数合理"""
        waveform = tf.random.normal([15000])  # 音频信号长度为 15000
        frame_length = 400
        frame_step = 400
        num_windows = 31

        result = get_windows(waveform, frame_length, frame_step, num_windows)

        # 验证输出形状
        expected_shape = (num_windows, frame_length, 2)  # 音频长度为15000，可以生成2个拼接窗
        self.assertEqual(result.shape, expected_shape)

    def test_boundary_case(self):
        """测试边界情况：音频长度刚好满足要求"""
        waveform = tf.random.normal([4000])  # 音频长度刚好满足要求
        frame_length = 400
        frame_step = 400
        num_windows = 31

        # 验证参数
        min_required_length = (num_windows - 1) * frame_step + frame_length
        min_required_length = tf.cast(min_required_length, tf.int32)
        self.assertEqual(min_required_length, 12400)  # 确保最小所需长度为 4000

        result = get_windows(waveform, frame_length, frame_step, num_windows)

        # 验证输出形状
        expected_shape = (num_windows, frame_length, 1)  # 只能生成1个拼接窗
        self.assertEqual(result.shape, expected_shape)

    # def test_insufficient_length(self):
    #     """测试音频长度不足的情况"""
    #     waveform = tf.random.normal([3000])  # 音频长度不足
    #     frame_length = 400
    #     frame_step = 400
    #     num_windows = 10
    #
    #     # 使用 unittest 提供的 assertRaises 方法来捕获错误
    #     with self.assertRaises(tf.errors.InvalidArgumentError):
    #         get_windows(waveform, frame_length, frame_step, num_windows)

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

    # def test_extreme_case(self):
    #     """测试极端情况：frame_length 或 frame_step 非常大"""
    #     waveform = tf.random.normal([1000])  # 音频信号长度为 1000
    #     frame_length = 1000
    #     frame_step = 1000
    #     num_windows = 10
    #
    #     with self.assertRaises(tf.errors.InvalidArgumentError):
    #         get_windows(waveform, frame_length, frame_step, num_windows)


if __name__ == '__main__':
    tf.test.main()
