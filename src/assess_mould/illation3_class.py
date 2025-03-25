import random
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import numpy as np

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_dir = Path("D:/PycharmProjects/wark_by_voice/dev_sample/1_wake")
time_step = 0.4
max_duration = 1
batch_size = 30  # 每页曲线数量

# 获取所有文件列表
file_list = list(data_dir.glob('*.wav'))
total_files = len(file_list)
random.shuffle(file_list)       # 打乱文件

# 加载导出的模型（包含预处理）
model_path = "D:/PycharmProjects/wark_by_voice/saved"
loaded_model = tf.saved_model.load(model_path)

# 分页处理
for page_idx in range(0, min(total_files, 400), batch_size):
    # 创建新画布
    plt.figure(figsize=(18, 10))  # 适当增加画布宽度

    # 获取当前页文件
    current_files = file_list[page_idx: page_idx + batch_size]

    # 绘制当前页的曲线
    for i, file_path in enumerate(current_files, 1):
        # 调用模型处理
        input_data = tf.constant(str(file_path), dtype=tf.string)
        output = loaded_model(input_data)

        # 获取预测数据
        predictions = output['predictions'].numpy()
        print(predictions)

        # 生成时间轴并截断
        time_points = np.arange(len(predictions)) * time_step
        truncate_idx = len(time_points) if np.all(time_points <= max_duration) else np.argmax(time_points > max_duration)

        # 绘制曲线（使用颜色映射）
        color = plt.cm.tab20(i % 20)  # 使用20种循环颜色
        plt.plot(time_points[:truncate_idx],
                 predictions[:truncate_idx],
                 alpha=0.7,
                 linewidth=1,
                 color=color,
                 label=f'{i}. {file_path.stem[:12]}')  # 添加序号防止混淆

    # 公共元素设置
    if page_idx == 0:  # 只在第一页设置阈值标签
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.2, label='阈值')
    else:
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.2)
    plt.title(f'模型预测结果对比（第 {page_idx // batch_size + 1} 页）', fontsize=14)
    plt.xlabel('时间（秒）', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.xlim(0, max_duration)

    plt.ylim(-0.05, 1.05)


    # 智能图例布局
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.0,
        fontsize=8,
        ncol=2,
        title=f'文件序列（第 {page_idx // batch_size + 1} 页）',
        title_fontsize=9,
        framealpha=0.5
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

# 打印汇总信息
print(f'共处理 {total_files} 个文件')
print(f'生成 {((total_files - 1) // batch_size) + 1} 张图表')




#

