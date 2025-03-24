import os
from pydub import AudioSegment

def calculate_average_duration(folder_path):
    # 初始化总时长和文件计数器
    total_duration = 0
    file_count = 0

    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是音频文件（这里简单地通过文件扩展名来判断）
        if filename.lower().endswith(('.mp3', '.wav', '.ogg', '.flac', '.aac')):
            # 构造完整的文件路径
            file_path = os.path.join(folder_path, filename)
            try:
                # 使用pydub加载音频文件
                audio = AudioSegment.from_file(file_path)
                # 累加音频时长（单位为毫秒）
                total_duration += len(audio)
                # 增加文件计数
                file_count += 1
            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")

    # 如果没有找到任何音频文件，返回0
    if file_count == 0:
        print("未找到音频文件。")
        return 0

    # 计算平均时长（单位为秒）
    average_duration = total_duration / 1000 / file_count
    return average_duration

# 指定音频文件所在的文件夹路径
folder_path = "D:\PycharmProjects\wark_by_voice\原素材\缓存器\唤醒词\\1_wake_mini1\临时效果1_wake"
average_duration = calculate_average_duration(folder_path)

if average_duration > 0:
    print(f"所有音频文件的平均时长为：{average_duration:.2f} 秒")