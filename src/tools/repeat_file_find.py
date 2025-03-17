import os

def compare_folders(folder1, folder2):
    # 获取两个文件夹中的文件名列表
    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = set(os.listdir(folder2))

    # 找出重复的文件名
    common_files = files_in_folder1.intersection(files_in_folder2)

    # 如果有重复的文件名，打印出来
    if common_files:
        print("以下文件名在两个文件夹中都存在：")
        for file in common_files:
            print(file)
    else:
        print("两个文件夹中没有重复的文件名。")

# 示例用法
folder1 = "D:\PycharmProjects\wark_by_voice\原素材\LibriSpeech_son_train"
folder2 = "D:\PycharmProjects\wark_by_voice\原素材\LibriSpeech_son_dev"

compare_folders(folder1, folder2)