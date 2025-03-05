import os
import shutil
import random

def move_wav_files_randomly(parent_folder, new_folder_name, percent):
    # 创建一个新的文件夹，如果不存在的话
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"创建新文件夹：{new_folder_path}")
    else:
        # 清空文件夹内容
        for filename in os.listdir(new_folder_path):
            file_path = os.path.join(new_folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子文件夹
        print("代存目标文件夹内容已经存在，已经预先清空")

    # 收集所有.wav文件的完整路径
    all_wav_files = []
    for root, dirs, files in os.walk(parent_folder):
        wav_files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
        all_wav_files.extend(wav_files)

    # 计算需要随机选取的文件数量
    num_to_select = int(len(all_wav_files) * percent)
    if num_to_select > 0:
        selected_files = set()  # 用于记录已经选择的文件路径
        while len(selected_files) < num_to_select:
            # 随机选择一个文件
            random_file = random.choice(all_wav_files)
            file_name = os.path.basename(random_file)
            destination_path = os.path.join(new_folder_path, file_name)

            # 检查目标文件夹是否已经存在同名文件
            if not os.path.exists(destination_path):
                selected_files.add(random_file)  # 记录已选择的文件
                shutil.copy2(random_file, destination_path)
                # print(f"复制文件：{random_file} -> {destination_path}")
            else:
                print(f"跳过文件（目标文件夹中已存在）：{file_name}")

    print(f"随机选取并复制.wav文件完成！总共复制了 {len(selected_files)} 个文件。")

# 使用示例
# parent_folder = r"D:\PycharmProjects\wark_by_voice\简略版数据集-420条语音\mobvoi_hotword_dataset"  #   验证
# parent_folder = r"D:\PycharmProjects\wark_by_voice\we_train\train\SPEECHDATA\wav"  # 替换为你的父文件夹路径
parent_folder = r"D:\BaiduNetdiskDownload\环境场景音效包\【环境】音效"
# parent_folder = r"D:\PycharmProjects\wark_by_voice\AISHELL-WakeUp-1-sample\SPEECHDATA\speech\wav\1_wake_words"
parent_folder = r"D:\PycharmProjects\wark_by_voice\sample_train\大数据量预存\非唤醒词"

new_folder_name = "wav_files"  # 新文件夹的名称
percent = 0.2  # 随机选取的百分比，例如0.5表示选取50%

move_wav_files_randomly(parent_folder, new_folder_name, percent)

