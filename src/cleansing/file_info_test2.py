import os
import wave
import logging
from pydub import AudioSegment

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_wav_file(file_path, target_rate=16000, target_bits=16):
    """检查WAV文件是否符合指定格式"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth() * 8  # 转换为位数
            return (sample_rate == target_rate and sample_width == target_bits), {
                "sample_rate": sample_rate,
                "sample_width": sample_width,
                "num_channels": wav_file.getnchannels(),
                "num_frames": wav_file.getnframes()
            }
    except Exception as e:
        logging.error(f"文件损坏或无法读取: {file_path} - {str(e)}")
        return False, None


def resample_audio(file_path, target_rate=16000, target_bits=16):
    """重采样并转换位深"""
    try:
        audio = AudioSegment.from_wav(file_path)
        # 调整采样率
        resampled_audio = audio.set_frame_rate(target_rate)
        # 调整位深
        target_sample_width = target_bits // 8
        return resampled_audio.set_sample_width(target_sample_width)
    except Exception as e:
        logging.error(f"文件处理失败: {file_path} - {str(e)}")
        return None


def scan_and_process(directory, target_rate=16000, target_bits=16):
    """扫描并处理目录中的音频文件"""
    deleted_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)

                # 检查文件格式
                is_valid, details = check_wav_file(file_path, target_rate, target_bits)

                if not is_valid and details:
                    # 处理位深不足的情况
                    if details["sample_width"] < target_bits:
                        try:
                            os.remove(file_path)
                            deleted_files.append(file_path)
                            logging.warning(f"已删除低质量文件: {file_path} (位深: {details['sample_width']}位)")
                        except Exception as e:
                            logging.error(f"删除失败: {file_path} - {str(e)}")

                    # 处理其他需要重采样的情形
                    else:
                        if (processed_audio := resample_audio(file_path, target_rate, target_bits)):
                            try:
                                processed_audio.export(file_path, format="wav")
                                logging.info(f"已更新文件: {file_path}")
                            except Exception as e:
                                logging.error(f"保存失败: {file_path} - {str(e)}")
    return deleted_files


def main():
    # 设置目标目录
    data_dir = "D:\PycharmProjects\wark_by_voice\原素材\缓存器\非唤醒词\car_sounds"

    if not os.path.exists(data_dir):
        logging.error("目录不存在！")
        return

    # 执行处理流程
    removed_files = scan_and_process(data_dir)
    print('处理为16000khz和16bit完毕')

    # 打印删除报告
    print("\n删除报告:")
    if removed_files:
        for f in removed_files:
            print(f"▸ {f}")
        print(f"\n共删除 {len(removed_files)} 个低质量文件")
    else:
        print("没有需要删除的文件")


if __name__ == '__main__':
    main()