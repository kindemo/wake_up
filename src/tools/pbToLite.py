import logging

import tensorflow as tf

# 加载 SavedModel 格式的模型
model = tf.saved_model.load("D:/PycharmProjects/wark_by_voice/saved")
print(model.signatures.keys())

# 获取模型签名（默认签名）
# concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func = model.signatures["wave_input"]    # 假设模型有 "mfcc_input" 方法
# 设置转换器
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# 启用 Flex 模式
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用 TensorFlow Lite 内置操作
    tf.lite.OpsSet.SELECT_TF_OPS    # 启用 TensorFlow Select 操作
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter._experimental_default_to_single_thread_inference = True

# 可选：启用量化优化（减少模型大小，可能损失精度）
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.experimental_new_converter = True  # 确保启用新转换器
converter._experimental_allow_all_select_tf_ops = True  # 允许所有TensorFlow操作（已弃用，但某些旧版本需要）
logging.basicConfig(level=logging.INFO)

# 转换为 TFLite 模型
tflite_model = converter.convert()

# 保存为 .tflite 文件
with open("model_wave3.tflite", "wb") as f:
    f.write(tflite_model)







# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# print("Input details:", input_details[0]["dtype"], input_details[0]["shape"])
# print("Output details:", output_details[0]["dtype"], output_details[0]["shape"])