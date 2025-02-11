import tensorflow as tf
import matplotlib.pyplot as plt
from func import get_spectrogram,squeeze
from pydub import AudioSegment

# 加载模型
imported = tf.saved_model.load("D:\\PycharmProjects\\wark_by_voice\\saved")

# 准备输入数据
audio_file_path = "D:\\PycharmProjects\\wark_by_voice\\verify\\yang_21_1.wav"
audio = tf.io.read_file(audio_file_path)
audio, _ = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=16000)
audio = tf.squeeze(audio, axis=-1)
audio = audio[tf.newaxis, :]  # 增加批次维度

# 使用模型进行预测
predictions = imported(audio)
print(predictions)

# 提取预测结果
class_ids = predictions['class_ids'].numpy()
class_names = predictions['class_names'].numpy()
class_names = [name.decode('utf-8') for name in class_names]
prediction_probabilities = predictions['predictions'].numpy()

# 打印预测的类别名称和概率
print("Predicted class names:", class_names)
print("Prediction probabilities:", prediction_probabilities[0])

# 绘制预测的概率分布
plt.bar(class_names, prediction_probabilities[0])
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Prediction Results')
plt.show()