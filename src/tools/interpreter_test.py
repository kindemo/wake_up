import tensorflow as tf
import numpy as np

# 测试TFLite模型是否正常工作
interpreter = tf.lite.Interpreter("model_wave2.tflite")

interpreter.allocate_tensors()

# 准备与Java端相同的输入（12800个随机样本）
test_input = np.random.randn(1, 12800).astype(np.float32)
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], test_input)

# 触发推理（此处应复现Java端的错误）
interpreter.invoke()  # 如果此处报错，说明模型转换存在问题