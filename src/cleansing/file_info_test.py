import os
import wave
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_wav_file(file_path, target_rate=16000, target_bits=16):
    """
    检查单个 WAV 文件是否符合指定的采样率和位深。
    如果文件不符合要求，返回 False 和详细信息；否则返回 True 和 None。
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频文件的基本信息
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth() * 8  # 转换为位数
            num_channels = wav_file.getnchannels()
            num_frames = wav_file.getnframes()

            # 检查是否符合要求
            if sample_rate != target_rate or sample_width != target_bits:
                return False, {
                    "file_path": file_path,
                    "sample_rate": sample_rate,
                    "sample_width": sample_width,
                    "num_channels": num_channels,
                    "num_frames": num_frames
                }
            else:
                return True, None
    except (wave.Error, EOFError, Exception) as e:
        # 如果文件损坏或无法读取，记录错误并返回 False
        logging.error(f"Error reading file {file_path}: {e}")
        return False, None

def scan_wav_files(directory, target_rate=16000, target_bits=16):
    """
    遍历指定目录及其子目录中的所有 WAV 文件，并检查每个文件是否符合指定格式。
    返回一个包含不符合要求的文件及其详细信息的列表。
    """
    mismatched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                is_valid, details = check_wav_file(file_path, target_rate, target_bits)
                if not is_valid:
                    mismatched_files.append(details)
    return mismatched_files

def main():
    # 指定要扫描的目录
    data_directory = "D:\PycharmProjects\wark_by_voice\原素材\HuanJing_train"
    if not os.path.isdir(data_directory):
        logging.error("指定的路径不是一个有效的目录！")
        return

    # 扫描并检查 WAV 文件
    mismatched_files = scan_wav_files(data_directory, target_rate=16000, target_bits=16)
    if mismatched_files:
        logging.warning(f"发现 {len(mismatched_files)} 个不符合要求的 WAV 文件：")
        for details in mismatched_files:
            logging.warning(f"文件路径：{details['file_path']}")
            logging.warning(f"采样率：{details['sample_rate']} Hz")
            logging.warning(f"位深：{details['sample_width']} bit")
            logging.warning(f"声道数：{details['num_channels']}")
            logging.warning(f"帧数：{details['num_frames']}")
            logging.warning("-" * 50)
    else:
        logging.info("所有 WAV 文件均符合要求（16000 Hz, 16-bit）。")

if __name__ == '__main__':
    main()