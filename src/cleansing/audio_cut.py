import os
import wave
import random
import shutil

def process_audio_files(input_folder, output_folder):
    """
    遍历指定文件夹中的所有.wav文件，随机裁切5秒的音频片段。
    如果音频长度不足5秒，则直接复制到输出文件夹。
    裁切后的音频保存到新的文件夹中。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)  # 删除旧的输出文件夹
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开音频文件
                with wave.open(input_path, 'rb') as wav_file:
                    # 获取音频参数
                    n_channels, sampwidth, framerate, n_frames, comptype, compname = wav_file.getparams()
                    duration = n_frames / float(framerate)  # 计算音频时长

                    # 如果音频长度大于5秒，则随机裁切5秒的片段
                    if duration >= 5:
                        start_frame = random.randint(0, int((duration - 5) * framerate))
                        end_frame = start_frame + int(5 * framerate)

                        # 创建一个新的音频文件
                        with wave.open(output_path, 'wb') as output_wav:
                            output_wav.setparams(wav_file.getparams())
                            wav_file.setpos(start_frame)
                            audio_data = wav_file.readframes(end_frame - start_frame)
                            output_wav.writeframes(audio_data)
                        print(f"Processed and saved: {output_path}")
                    # 如果音频长度不足5秒，则直接复制到输出文件夹
                    else:
                        pass
                        # shutil.copy(input_path, output_path)
                        # print(f"Audio too short, copied directly: {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    print("Processing complete.")


# 示例用法
if __name__ == "__main__":
    input_folder = r"D:\PycharmProjects\wark_by_voice\原素材\aishell_train\speech"  # 替换为你的输入文件夹路径
    output_folder = r"D:\PycharmProjects\wark_by_voice\原素材\aishell_train\speech\cut"  # 替换为你的输出文件夹路径
    process_audio_files(input_folder, output_folder)