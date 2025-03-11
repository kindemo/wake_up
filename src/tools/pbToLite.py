import tensorflow as tf

# 加载 SavedModel 格式的模型
model = tf.saved_model.load("D:/PycharmProjects/wark_by_voice/saved")
print(model.signatures.keys())

# 获取模型签名（默认签名）
# concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func = model.signatures["mfcc_input"]    # 假设模型有 "mfcc_input" 方法
# 设置转换器
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# 可选：启用量化优化（减少模型大小，可能损失精度）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 转换为 TFLite 模型
tflite_model = converter.convert()

# 保存为 .tflite 文件
with open("model.tflite", "wb") as f:
    f.write(tflite_model)



# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# print("Input details:", input_details[0]["dtype"], input_details[0]["shape"])
# print("Output details:", output_details[0]["dtype"], output_details[0]["shape"])