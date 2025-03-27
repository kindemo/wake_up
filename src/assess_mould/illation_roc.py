import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.python.ops.ragged.ragged_array_ops import expand_dims

from src.data_loader import create_interleaved_dataset, load_dataset_dev, load_dataset_train
from src.preprocessing.Pretreatment import load_and_split_audio, preprocess_dataset

# 配置参数
f_block = 16
num_win = 56
batch_size = 32

# 加载模型
model = tf.saved_model.load("D:/PycharmProjects/wark_by_voice/saved")
data_dev_dir = "D:/PycharmProjects/wark_by_voice/dev_sample"

dev_file_paths, labels_dev = load_dataset_dev(data_dev_dir)  # 加载模型训练文件
dataset_dev = create_interleaved_dataset(dev_file_paths, labels_dev, block_size=f_block)  # 路径和标签绑定的dataset dev
dataset_dev, labels_dev = preprocess_dataset(dataset_dev, num_win=56)  # dev 加載自定義預處理

# 执行预测
all_pred = []
all_labels = []

for batch in dataset_dev:
    features, labels = batch
    features = expand_dims(features, axis=0)
    # print("features.shape, labels.shape",features.shape, labels.shape)
    pred = model(features)['predictions']

    all_pred.append(pred)
    all_labels.append(labels)



# 将概率转换为二分类预测
threshold = 0.5
binary_pred = np.where(np.array(all_pred) >= threshold, 1, 0)
all_labels = np.where(np.array(all_labels) >= threshold, 1, 0)

# 初始化四个指标
TP = 0  # 真正例
TN = 0  # 真负例
FP = 0  # 假正例
FN = 0  # 假负例

# 遍历每个样本计算指标
for pred, label in zip(binary_pred, all_labels):
    if label == 1:
        if pred == 1:
            TP += 1
        else:
            FN += 1
    else:
        if pred == 1:
            FP += 1
        else:
            TN += 1

# 组装混淆矩阵
confusion_matrix = [
    [TN, FP],
    [FN, TP]
]

print("Confusion Matrix:")
print(f"[[{TN}  {FP}]")
print(f" [{FN}  {TP}]]")


# 正确计算混淆矩阵
cm = confusion_matrix

# 可视化设置
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm,
                 annot=True,
                 fmt='d',  # 显示整数
                 cmap='Blues',  # 颜色主题
                 cbar=False,
                 linewidths=1,
                 linecolor='black')

# 坐标轴标签
ax.set(xlabel='Predicted Label',
       ylabel='True Label',
       xticklabels=['0', '1'],
       yticklabels=['0', '1'])

# 标题
plt.title('Confusion Matrix', fontsize=14, pad=20)

# 显示图形
plt.show()



# 将预测结果和标签转换为numpy数组
# all_pred_probs = np.concatenate([p.reshape(-1) for p in all_pred])  # 确保每个元素是一维数组
# all_labels = np.concatenate([l.reshape(-1) for l in all_labels]).astype(np.int32)  # 确保标签也是一维


# 计算其他评估指标
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
f_kws = FP /(TP + FP) if (TP + FP) != 0 else 0

print("\nAdditional Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"KWS rate: {recall:.4f}")
print(f"wrong KWS rate: {f_kws:.4f}")
print(f"F1 Score: {f1:.4f}")


# 绘制评估指标柱状图
metrics = ['Accuracy', 'Precision', 'Recall(wake)','wrong wake','F1 Score']
values = [accuracy, precision, recall, f_kws, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'pink'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')

# 在柱子上方添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.show()

# 确保为一维
all_pred_1d = np.array(all_pred).flatten()
all_labels = all_labels.flatten()

# 计算ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_pred_1d)
roc_auc = roc_auc_score(all_labels, all_pred_1d)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()