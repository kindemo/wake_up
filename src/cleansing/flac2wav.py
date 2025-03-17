from pydub import AudioSegment
import os

# 设置源文件夹路径（包含FLAC文件的文件夹）
source_folder = "D:\PycharmProjects\wark_by_voice\原素材\libriSpeech_son"  # 替换为你的FLAC文件所在文件夹路径
# 设置目标文件夹路径（转换后的WAV文件将保存在此）
target_folder = "D:\PycharmProjects\wark_by_voice\原素材\libriSpeech_son\kresult"  # 替换为你的目标文件夹路径

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for file_name in os.listdir(source_folder):
    # 检查文件扩展名是否为.flac
    if file_name.endswith(".flac"):
        # 构造完整的文件路径
        flac_file_path = os.path.join(source_folder, file_name)
        # 构造目标文件名（替换扩展名为.wav）
        wav_file_name = file_name.replace(".flac", ".wav")
        wav_file_path = os.path.join(target_folder, wav_file_name)

        # 使用pydub加载FLAC文件并导出为WAV
        audio = AudioSegment.from_file(flac_file_path, format="flac")
        audio.export(wav_file_path, format="wav")
        print(f"Converted {flac_file_path} to {wav_file_path}")

print("All FLAC files have been converted to WAV format.")