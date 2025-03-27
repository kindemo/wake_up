import os
import wave
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_wav_file(file_path):
    """
    检查单个 WAV 文件是否可以正常解码。
    如果文件损坏或无法解码，返回 False，否则返回 True。
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取一些基本信息以确保文件可以正常读取
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            # 如果所有信息都能正常获取，说明文件是好的
            return True
    except (wave.Error, EOFError, Exception) as e:
        # 如果在打开或读取过程中抛出异常，记录日志并返回 False
        logging.error(f"Error reading file {file_path}: {e}")
        return False

def scan_wav_files(directory):
    """
    遍历指定目录及其子目录中的所有 WAV 文件，并检查每个文件是否损坏。
    返回一个包含损坏文件路径的列表。
    """
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                if not check_wav_file(file_path):
                    corrupted_files.append(file_path)
    return corrupted_files

def main():
    # 指定要扫描的目录

    data_directory = "D:\PycharmProjects\wark_by_voice\原素材\缓存器\非唤醒词\car_sounds"
    if not os.path.isdir(data_directory):
        logging.error("指定的路径不是一个有效的目录！")
        return

    # 扫描并检查 WAV 文件
    corrupted_files = scan_wav_files(data_directory)
    if corrupted_files:
        logging.warning(f"发现 {len(corrupted_files)} 个损坏的 WAV 文件：")
        for file in corrupted_files:
            logging.warning(f"损坏的文件：{file}")
    else:
        logging.info("未发现损坏的 WAV 文件。")

if __name__ == '__main__':
    main()