import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

# 配置文件路径
data_dir = Path("D:/PycharmProjects/wark_by_voice/dev_sample/0_non_wake")
model_path = "D:/PycharmProjects/wark_by_voice/saved"
output_excel_path = "D:/PycharmProjects/wark_by_voice/predictions.xlsx"

# 创建输出目录（如果不存在）
output_dir = Path(output_excel_path).parent
if not output_dir.exists():
    os.makedirs(output_dir)

# 加载TensorFlow模型
loaded_model = tf.saved_model.load(model_path)

# 获取并随机打乱音频文件列表
file_list = list(data_dir.glob('*.wav'))
random.shuffle(file_list)

# 时间轴配置
time_step = 0.4  # 时间间隔（秒）
max_duration = 1  # 最大持续时间（秒）
time_labels = [f"{t:.1f}s" for t in np.arange(0, max_duration + time_step, time_step)]

# 数据收集列表
data_records = []

# 处理每个音频文件
for file_path in file_list:
    # 模型推理
    input_tensor = tf.constant(str(file_path), dtype=tf.string)
    model_output = loaded_model(input_tensor)

    # 获取预测结果并展平
    predictions = model_output['predictions'].numpy().flatten()

    # 数据对齐处理
    aligned_predictions = np.full(len(time_labels), np.nan)  # 预填充NaN
    copy_length = min(len(predictions), len(time_labels))
    aligned_predictions[:copy_length] = predictions[:copy_length]

    # 创建数据记录
    record = {
        "文件名": file_path.stem,
    ** {time_labels[i]: aligned_predictions[i] for i in range(len(time_labels))}
    }
    data_records.append(record)

# 创建并转置DataFrame
df = pd.DataFrame(data_records).set_index('文件名')

# 导出到Excel
with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='预测结果', float_format="%.4f")

print(f"预测结果已导出至：{output_excel_path}")
print(f"数据维度：{df.shape}（{len(data_records)}个文件 × {len(time_labels)}个时间点）")