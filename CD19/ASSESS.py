# cd19
import os
import paddle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import paddle.nn.functional as F

plt.rcParams.update({
    'font.family': 'Arial',  # 使用Arial字体
    'font.size': 14,  # 基础字体大小
    'axes.titlesize': 18,  # 标题字体大小
    'axes.labelsize': 16,  # 坐标轴标签字体大小
    'xtick.labelsize': 14,  # x轴刻度字体大小
    'ytick.labelsize': 14,  # y轴刻度字体大小
    'legend.fontsize': 14,  # 图例字体大小
    'axes.grid': False,  # 默认不显示网格线
    'axes.edgecolor': 'black',  # 坐标轴边框颜色
    'savefig.dpi': 300  # 保存图片分辨率
})

# 设置路径
data_dir = '/home/aistudio/data/CD19'
save_dir = '/home/aistudio/work/cnn_lstm_CD19'
picture_dir = '/home/aistudio/work/cnn_lstm_CD19_PICTURE'
os.makedirs(picture_dir, exist_ok=True)

# 导入必要的类和函数
from paddle.vision.models import resnet18
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from PIL import Image


# 定义模型类
class MultiHeadAttentionLayer(nn.Layer):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.multi_head_attn = nn.MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads
        )

    def forward(self, lstm_out):
        lstm_out_trans = paddle.transpose(lstm_out, perm=[1, 0, 2])
        attn_outputs = self.multi_head_attn(
            query=lstm_out_trans,
            key=lstm_out_trans,
            value=lstm_out_trans
        )

        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
        else:
            attn_output = attn_outputs

        attn_output = paddle.transpose(attn_output, perm=[1, 0, 2])
        context = paddle.sum(attn_output, axis=0)
        return context


class CNNLSTM(nn.Layer):
    def __init__(self, cnn_output_size=512, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.cnn_bn = nn.BatchNorm1D(cnn_output_size)
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction='bidirectional',
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = MultiHeadAttentionLayer(hidden_size * 2, num_heads=4)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = paddle.reshape(x, [batch_size * seq_len, c, h, w])
        cnn_features = self.cnn(x)
        cnn_features = self.cnn_bn(cnn_features)
        cnn_features = paddle.reshape(cnn_features, [batch_size, seq_len, -1])
        cnn_features = paddle.transpose(cnn_features, perm=[1, 0, 2])
        lstm_out, _ = self.lstm(cnn_features)
        context = self.attention(lstm_out)
        context = self.fc_dropout(context)
        output = self.fc(context)
        return output


# 数据集类定义
class ConcentrationDataset(Dataset):
    def __init__(self, data_dir, phase, class_names, class_values_scaled, scaler, transform=None, sequence_length=5):
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequences = []
        self.scaler = scaler
        self.class_names = class_names
        self.class_values_scaled = class_values_scaled

        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")

        total_sequences = 0
        for i, cls in enumerate(class_names):
            class_dir = os.path.join(data_dir, cls)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory does not exist - {class_dir}")
                continue

            subdir = 'training' if phase == 'train' else 'validation'
            data_subdir = os.path.join(class_dir, subdir)

            if not os.path.exists(data_subdir):
                print(f"Warning: {phase} subdirectory does not exist - {data_subdir}")
                continue

            image_files = [f for f in os.listdir(data_subdir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.gif'))]

            if not image_files:
                print(f"Warning: No image files found in {data_subdir}")
                continue

            image_files = sorted(image_files)
            if len(image_files) < sequence_length:
                print(f"Warning: Insufficient images in {data_subdir} to form sequences")
                continue

            for j in range(len(image_files) - sequence_length + 1):
                sequence = image_files[j:j + sequence_length]
                sequence_paths = [os.path.join(data_subdir, f) for f in sequence]
                self.sequences.append((sequence_paths, class_values_scaled[i], float(cls)))
                total_sequences += 1

        print(f"Successfully loaded {phase} dataset with {total_sequences} sequences")
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences found in {phase} dataset")

    def __getitem__(self, idx):
        img_paths, label_scaled, label_original = self.sequences[idx]
        imgs = []

        for path in img_paths:
            try:
                img = Image.open(path).convert('RGB')
                if img.format == 'GIF':
                    img.seek(0)
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except Exception as e:
                print(f"Error: Failed to load image {path}: {e}")
                img = Image.new('RGB', (150, 150), color='black')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

        imgs = paddle.stack(imgs, axis=0)
        return imgs, paddle.to_tensor(label_scaled, dtype='float32').reshape([-1]), label_original

    def __len__(self):
        return len(self.sequences)

    def inverse_transform(self, y_pred_scaled):
        y_pred_log = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_original = np.exp(y_pred_log)
        return y_pred_original


# 数据加载和预测
class_names = ['18000000', '9000000', '1800000', '360000', '90000', '18000', '9000', '4500']  # 更新为新类别
class_values = np.array([float(cls) for cls in class_names])
num_classes = len(class_names)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
class_values_log = [np.log(float(cls)) for cls in class_names]
class_values_scaled = scaler.fit_transform(np.array(class_values_log).reshape(-1, 1)).flatten()

from paddle.vision.transforms import Compose, Resize, ToTensor, Normalize

valid_transform = Compose([
    Resize((150, 150)),
    ToTensor(),
    Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 加载新模型
model = CNNLSTM(cnn_output_size=512, hidden_size=128, num_layers=2, dropout=0.3)
model_state_dict = paddle.load(os.path.join(save_dir, 'best_cnn_lstm_model.pdparams'))
model.set_state_dict(model_state_dict)
model.eval()


class FullDataset(ConcentrationDataset):
    def __init__(self, data_dir, class_names, class_values_scaled, scaler, transform=None, sequence_length=5):
        super().__init__(data_dir, 'train', class_names, class_values_scaled, scaler, transform, sequence_length)
        valid_data = ConcentrationDataset(data_dir, 'valid', class_names, class_values_scaled, scaler, transform,
                                          sequence_length)
        self.sequences.extend(valid_data.sequences)
        print(f"Total sequences after merging: {len(self.sequences)}")


full_dataset = FullDataset(
    data_dir=data_dir,
    class_names=class_names,
    class_values_scaled=class_values_scaled,
    scaler=scaler,
    transform=valid_transform,
    sequence_length=5
)
full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=2)

all_preds = []
all_labels = []
all_probs = []
all_errors = []  # 存储每个样本的误差

with paddle.no_grad():
    for imgs, labels_scaled, labels_original_batch in full_loader:
        outputs_scaled = model(imgs)
        outputs_original = full_dataset.inverse_transform(outputs_scaled.numpy().flatten())

        for pred_val, true_val in zip(outputs_original, labels_original_batch):
            true_idx = class_names.index(str(int(true_val)))
            all_labels.append(true_idx)

            diffs = np.abs(class_values - pred_val)
            pred_idx = np.argmin(diffs)
            all_preds.append(pred_idx)

            sigma = 1000000
            probs = np.exp(-diffs ** 2 / (2 * sigma ** 2))
            probs = probs / np.sum(probs)
            all_probs.append(probs)

            # 计算误差 (真实值 - 预测值)
            error = float(true_val) - pred_val
            all_errors.append(error)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_errors = np.array(all_errors)

valid_classes = np.unique(all_labels)
missing_classes = set(range(num_classes)) - set(valid_classes)
if missing_classes:
    print(f"⚠️ Warning: Some classes are missing in the data: {missing_classes}")

# 构建类别误差字典
class_errors = {class_names[i]: [] for i in range(num_classes)}
for true_idx, error in zip(all_labels, all_errors):
    class_errors[class_names[true_idx]].append(error)

# 1. 混淆矩阵
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', cbar=True,
            linewidths=0.5,
            annot_kws={"size": 14})
plt.title('Confusion Matrix', pad=20)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'confusion_matrix.png'), bbox_inches='tight')
plt.show()

# 2. 归一化混淆矩阵
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
cm_normalized[np.isnan(cm_normalized)] = 0
sns.heatmap(cm_normalized, annot=True, fmt='.2f',
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', cbar=True,
            linewidths=0.5,
            annot_kws={"size": 14})
plt.title('Normalized Confusion Matrix', pad=20)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'normalized_confusion_matrix.png'), bbox_inches='tight')
plt.show()

# 3. F1 Score
f1_scores = f1_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0)
plt.figure(figsize=(12, 8))
sns.barplot(x=list(range(num_classes)), y=f1_scores,
            color='#4287f5',
            edgecolor='black')
plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
plt.title('F1 Score per Class', pad=20)
plt.ylabel('F1 Score')
plt.xlabel('Class')
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.3)
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'f1_score.png'), bbox_inches='tight')
plt.show()

# 4. PR Curve
y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', n_colors=num_classes)
for i in range(num_classes):
    if i in valid_classes:
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        plt.plot(recall, precision, lw=2,
                 color=colors[i],
                 label=f'{class_names[i]} (AP={ap:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve", pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           frameon=True,
           edgecolor='black')
plt.grid(linestyle='--', alpha=0.3)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'pr_curve.png'), bbox_inches='tight')
plt.show()

# 5. 类别分布
plt.figure(figsize=(12, 8))
sns.countplot(x=all_labels, order=range(num_classes),
              color='#4287f5',
              edgecolor='black')
plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
plt.title('Class Distribution in All Data', pad=20)
plt.ylabel('Number of Samples')
plt.xlabel('Class')
plt.grid(axis='y', linestyle='--', alpha=0.3)
for i, v in enumerate(np.bincount(all_labels)):
    plt.text(i, v + 5, f'{v}', ha='center', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'class_distribution.png'), bbox_inches='tight')
plt.show()

# 6. 类别误差箱线图
plt.figure(figsize=(10, 8))
plt.boxplot(class_errors.values(), labels=class_errors.keys())
plt.ylabel('Error (True - Predicted)')
plt.title('Class-wise Error Distribution')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, 'class_error_boxplot.png'), bbox_inches='tight')
plt.show()

print(f"所有图表已保存至: {picture_dir}")