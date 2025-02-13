import os
import shutil


def move_wav_files(parent_folder, new_folder_name):
    # 创建一个新的文件夹，如果不存在的话
    new_folder_path = os.path.join(parent_folder, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f"创建新文件夹：{new_folder_path}")

    # 遍历父文件夹中的所有子文件夹
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith('.wav'):  # 检查文件扩展名是否为.wav
                source_path = os.path.join(root, file)
                destination_path = os.path.join(new_folder_path, file)

                # 如果目标文件夹中已经存在同名文件，则重命名新文件
                if os.path.exists(destination_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_path):
                        new_file_name = f"{base}_{counter}{extension}"
                        destination_path = os.path.join(new_folder_path, new_file_name)
                        counter += 1

                # 移动文件
                shutil.move(source_path, destination_path)
                # print(f"移动文件：{source_path} -> {destination_path}")

    print("所有.wav文件已移动完成！")


# 使用示例
parent_folder = r"D:\PycharmProjects\wark_by_voice\we_train\train\SPEECHDATA\wav"  # 替换为你的父文件夹路径
new_folder_name = "wav_files"  # 新文件夹的名称
move_wav_files(parent_folder, new_folder_name)