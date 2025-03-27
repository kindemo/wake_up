import os
import random
import math
from pydub import AudioSegment


def mix_audio(folder_a, folder_b, output_folder, ratio_a=0.7, output_format="wav"):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹B中的所有音频文件
    b_files = []
    for f in os.listdir(folder_b):
        if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac')):
            b_files.append(os.path.join(folder_b, f))
    if not b_files:
        raise ValueError("文件夹B中没有音频文件")

    # 计算增益值（dB）
    gain_a_db = 20 * math.log10(ratio_a)
    gain_b_db = 20 * math.log10(1 - ratio_a)

    # 处理文件夹A中的每个文件
    for a_file in os.listdir(folder_a):
        if not a_file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac')):
            continue

        # 读取原始音频A
        a_path = os.path.join(folder_a, a_file)
        try:
            audio_a = AudioSegment.from_file(a_path)
        except:
            print(f"无法读取文件: {a_path}，跳过")
            continue

        # 随机选择B文件夹中的音频
        b_path = random.choice(b_files)
        try:
            audio_b = AudioSegment.from_file(b_path)
        except:
            print(f"无法读取文件: {b_path}，跳过")
            continue

        # 统一为单声道（可选）
        # audio_a = audio_a.set_channels(1)
        # audio_b = audio_b.set_channels(1)

        # 获取时长（毫秒）
        len_a = len(audio_a)
        len_b = len(audio_b)

        # 处理音频B
        if len_b > len_a:
            # 随机截取与A相同长度的片段
            start_pos = random.randint(0, len_b - len_a)
            audio_b = audio_b[start_pos: start_pos + len_a]
        else:
            # 保持B原有长度
            pass

        # 调整音量
        adjusted_a = audio_a.apply_gain(gain_a_db)
        adjusted_b = audio_b.apply_gain(gain_b_db)

        # 混合音频
        if len(adjusted_b) >= len(adjusted_a):
            # B长度 >= A长度，直接全长度混合
            mixed = adjusted_a.overlay(adjusted_b)
        else:
            # 随机选择混合起始位置
            max_start = len(adjusted_a) - len(adjusted_b)
            start_pos = random.randint(0, max_start)
            mixed = adjusted_a.overlay(adjusted_b, position=start_pos)

        # 保存结果
        output_path = os.path.join(output_folder, a_file)
        mixed.export(output_path, format=output_format)
        print(f"已处理: {a_file}")


if __name__ == "__main__":
    # 配置参数
    FOLDER_A = "D:\PycharmProjects\wark_by_voice\原素材\缓存器\唤醒词\\1_wake_mini1"
    FOLDER_B = "D:\PycharmProjects\wark_by_voice\原素材\缓存器\非唤醒词\car_sounds"
    OUTPUT_FOLDER = "D:\PycharmProjects\wark_by_voice\\train_sample\sofr_wake_0.8"
    MIX_RATIO = 0.8  # A音频比例

    # 执行混合
    mix_audio(FOLDER_A, FOLDER_B, OUTPUT_FOLDER, MIX_RATIO)