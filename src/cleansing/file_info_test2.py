import os
import wave
import logging
from pydub import AudioSegment

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

def resample_audio(file_path, target_rate=16000):
    """
    将音频文件重采样为指定的采样率（默认为 16000 Hz）。
    返回重采样后的音频对象。
    """
    try:
        # 使用 pydub 加载音频文件
        audio = AudioSegment.from_wav(file_path)
        # 重采样为指定的采样率
        resampled_audio = audio.set_frame_rate(target_rate)
        return resampled_audio
    except Exception as e:  # 捕获所有异常
        logging.error(f"Error during resampling file {file_path}: {e}")
        return None

def save_resampled_audio(resampled_audio, output_path):
    """
    保存重采样后的音频文件。
    """
    try:
        # 删除原始文件
        os.remove(output_path)
        # 保存重采样后的音频到原始文件路径
        resampled_audio.export(output_path, format="wav")
        logging.info(f"Saved resampled audio to {output_path}")
    except Exception as e:
        logging.error(f"Error saving resampled audio to {output_path}: {e}")

def scan_and_resample_wav_files(directory, target_rate=16000, target_bits=16):
    """
    遍历指定目录及其子目录中的所有 WAV 文件，检查并重采样不符合要求的文件。
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                is_valid, details = check_wav_file(file_path, target_rate, target_bits)
                if not is_valid:
                    logging.warning(f"File {file_path} does not meet the requirements:")
                    logging.warning(f"  Sample Rate: {details['sample_rate']} Hz")
                    logging.warning(f"  Sample Width: {details['sample_width']} bit")
                    logging.warning(f"  Channels: {details['num_channels']}")
                    logging.warning(f"  Frames: {details['num_frames']}")

                    # 重采样并保存
                    resampled_audio = resample_audio(file_path, target_rate)
                    if resampled_audio:
                        # 直接覆盖原始文件
                        save_resampled_audio(resampled_audio, file_path)
                else:
                    logging.info(f"File {file_path} meets the requirements.")

def main():
    # 指定要扫描的目录
    data_directory = "D:/PycharmProjects/wark_by_voice/sample_train/zhou_wake"
    if not os.path.isdir(data_directory):
        logging.error("指定的路径不是一个有效的目录！")
        return

    # 扫描并处理 WAV 文件
    scan_and_resample_wav_files(data_directory, target_rate=16000, target_bits=16)
    logging.info("处理完成。")

if __name__ == '__main__':
    main()