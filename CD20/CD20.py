# CD20
import os
import paddle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from paddle.io import Dataset, DataLoader
from paddle.vision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from paddle.vision.models import resnet18
import paddle.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# 设置路径和类别
data_dir = '/home/aistudio/data/CD20'
save_dir = '/home/aistudio/work/cnn_lstm_CD20'
os.makedirs(save_dir, exist_ok=True)

# 类别设置 - 使用对数转换
class_names = ['18000000', '9000000', '1800000', '360000', '72000', '36000', '18000', '9000']
class_values_log = [np.log(float(cls)) for cls in class_names]  # 对类别值取对数
num_classes = len(class_names)

# 创建标准化器并对对数化的类别值进行标准化
scaler = StandardScaler()
class_values_scaled = scaler.fit_transform(np.array(class_values_log).reshape(-1, 1)).flatten()


# 自定义变换
class CustomTransform:
    def __init__(self, func):
        self.func = func

    def __call__(self, img):
        return self.func(img)


def random_sharpness(img):
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(random.uniform(0.5, 2.0))


# 数据增强（简化版）
train_transform = Compose([
    Resize((180, 180)),
    RandomCrop((150, 150)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(10),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    CustomTransform(random_sharpness),
    ToTensor(),
    Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

valid_transform = Compose([
    Resize((150, 150)),
    ToTensor(),
    Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


# 序列数据集 - 增强版本
class ConcentrationDataset(Dataset):
    def __init__(self, data_dir, phase, class_names, class_values_scaled, scaler, transform=None, sequence_length=5):
        """
        初始化浓度数据集，支持序列数据处理
        """
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequences = []
        self.scaler = scaler
        self.class_names = class_names
        self.class_values_scaled = class_values_scaled

        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        total_sequences = 0
        for i, cls in enumerate(class_names):
            class_dir = os.path.join(data_dir, cls)
            if not os.path.exists(class_dir):
                print(f"警告: 类别目录不存在 - {class_dir}")
                continue

            subdir = 'training' if phase == 'train' else 'validation'
            data_subdir = os.path.join(class_dir, subdir)

            if not os.path.exists(data_subdir):
                print(f"警告: {phase}子目录不存在 - {data_subdir}")
                continue

            # 支持更多图像格式
            image_files = [f for f in os.listdir(data_subdir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.gif'))]

            if not image_files:
                print(f"警告: 在 {data_subdir} 中未找到图像文件")
                continue

            # 按文件名排序构建序列
            image_files = sorted(image_files)
            if len(image_files) < sequence_length:
                print(f"警告: {data_subdir} 中的图像数量不足以构成序列")
                continue

            # 构建滑动窗口序列
            for j in range(len(image_files) - sequence_length + 1):
                sequence = image_files[j:j + sequence_length]
                sequence_paths = [os.path.join(data_subdir, f) for f in sequence]
                self.sequences.append((sequence_paths, class_values_scaled[i], float(cls)))
                total_sequences += 1

        print(f"成功加载{phase}数据集，共{total_sequences}个序列")
        if len(self.sequences) == 0:
            raise ValueError(f"在{phase}数据集中未找到任何有效序列")

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
                print(f"错误: 加载图像 {path} 失败: {e}")
                # 加载失败时返回黑色图像
                img = Image.new('RGB', (150, 150), color='black')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)

        imgs = paddle.stack(imgs, axis=0)  # [sequence_length, 3, 150, 150]
        return imgs, paddle.to_tensor(label_scaled, dtype='float32').reshape([-1]), label_original

    def __len__(self):
        return len(self.sequences)

    def inverse_transform(self, y_pred_scaled):
        """将标准化的预测值转换回原始浓度尺度"""
        y_pred_log = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_pred_original = np.exp(y_pred_log)
        return y_pred_original


# 多头注意力层
class MultiHeadAttentionLayer(nn.Layer):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.multi_head_attn = nn.MultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads
        )

    def forward(self, lstm_out):
        # lstm_out形状：[seq_len, batch_size, hidden_size]
        # 多头注意力输入需 [batch_size, seq_len, hidden_size]，先转置
        lstm_out_trans = paddle.transpose(lstm_out, perm=[1, 0, 2])

        # 修改解包逻辑，接收所有返回值
        attn_outputs = self.multi_head_attn(
            query=lstm_out_trans,
            key=lstm_out_trans,
            value=lstm_out_trans
        )

        # 通常第一个返回值是注意力输出
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
        else:
            attn_output = attn_outputs

        # 转回 [seq_len, batch_size, hidden_size] 后求和
        attn_output = paddle.transpose(attn_output, perm=[1, 0, 2])
        context = paddle.sum(attn_output, axis=0)  # [batch_size, hidden_size]
        return context


# 双向LSTM + 多头注意力模型
class CNNLSTM(nn.Layer):
    def __init__(self, cnn_output_size=512, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # 去掉原ResNet18的全连接层

        # 添加批量归一化层
        self.cnn_bn = nn.BatchNorm1D(cnn_output_size)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction='bidirectional',
            dropout=dropout if num_layers > 1 else 0
        )

        # 多头注意力层
        self.attention = MultiHeadAttentionLayer(hidden_size * 2, num_heads=4)  # 双向LSTM输出维度加倍

        # 添加dropout层
        self.fc_dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, 1)  # 回归任务，输出1个预测值

    def forward(self, x):
        # x形状：[batch_size, sequence_length, 3, 150, 150]
        batch_size, seq_len, c, h, w = x.shape

        # CNN提取特征
        x = paddle.reshape(x, [batch_size * seq_len, c, h, w])
        cnn_features = self.cnn(x)  # 输出形状：[batch_size*seq_len, 512]

        # 应用批量归一化
        cnn_features = self.cnn_bn(cnn_features)

        cnn_features = paddle.reshape(cnn_features, [batch_size, seq_len, -1])  # [batch_size, seq_len, 512]

        # 调整为LSTM输入格式 [seq_len, batch_size, input_size]
        cnn_features = paddle.transpose(cnn_features, perm=[1, 0, 2])

        # LSTM处理
        lstm_out, _ = self.lstm(cnn_features)  # lstm_out形状：[seq_len, batch_size, hidden_size*2]

        # 注意力机制（多头）
        context = self.attention(lstm_out)  # [batch_size, hidden_size*2]

        # 应用dropout
        context = self.fc_dropout(context)

        # 最终预测
        output = self.fc(context)  # [batch_size, 1]
        return output


# 加载训练集和验证集
try:
    print("开始加载训练数据集...")
    train_dataset = ConcentrationDataset(
        data_dir=data_dir,
        phase='train',
        class_names=class_names,
        class_values_scaled=class_values_scaled,
        scaler=scaler,
        transform=train_transform,
        sequence_length=5  # 序列长度可调整
    )

    print("开始加载验证数据集...")
    valid_dataset = ConcentrationDataset(
        data_dir=data_dir,
        phase='valid',
        class_names=class_names,
        class_values_scaled=class_values_scaled,
        scaler=scaler,
        transform=valid_transform,
        sequence_length=5
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2)

except Exception as e:
    print(f"加载数据集时出错: {e}")
    # 打印详细的文件列表用于调试
    print("\n数据集中的文件详情:")
    if os.path.exists(data_dir):
        for cls in class_names:
            cls_dir = os.path.join(data_dir, cls)
            if os.path.exists(cls_dir):
                print(f"\n类别 {cls}:")
                for phase in ['training', 'validation']:
                    phase_dir = os.path.join(cls_dir, phase)
                    if os.path.exists(phase_dir):
                        files = os.listdir(phase_dir)
                        print(f"  {phase} 子目录 ({len(files)} 文件):")
                        for f in files[:5]:  # 显示前5个文件
                            print(f"    - {f}")
                        if len(files) > 5:
                            print(f"    ... 还有 {len(files) - 5} 个文件")
    else:
        print(f"数据目录 {data_dir} 不存在")
    raise

# 初始化模型、损失函数、优化器
model = CNNLSTM(cnn_output_size=512, hidden_size=128, num_layers=2, dropout=0.3)

# 使用梯度裁剪防止梯度爆炸
clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
criterion = nn.SmoothL1Loss()  # HuberLoss

# 训练循环
epochs = 100  # 先定义epochs变量
best_rmse = float('inf')

# 使用学习率调度器（余弦退火）
from paddle.optimizer.lr import CosineAnnealingDecay

lr_scheduler = CosineAnnealingDecay(
    learning_rate=0.001,
    T_max=epochs,  # 周期为训练轮数
    eta_min=0.00001  # 学习率下限
)

# 优化器换为AdamW，增加weight_decay
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    grad_clip=clip,
    weight_decay=0.001  # 权重衰减系数
)

# 训练过程记录
train_losses = []
valid_losses = []
valid_rmse_scores = []
valid_r2_scores = []
valid_mae_scores = []
valid_mape_scores = []

# 训练循环
epochs = 100
best_rmse = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, labels_scaled, labels_original in train_loader:
        outputs = model(imgs)
        loss = criterion(outputs, labels_scaled)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        train_loss += loss.numpy()[0] * imgs.shape[0]
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # 验证
    model.eval()
    valid_loss = 0.0
    preds_scaled = []
    truths_scaled = []
    preds_original = []
    truths_original = []

    with paddle.no_grad():
        for imgs, labels_scaled, labels_original_batch in valid_loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels_scaled)
            valid_loss += loss.numpy()[0] * imgs.shape[0]

            # 收集标准化的预测值和真实值
            preds_scaled.extend(outputs.cpu().numpy().flatten())
            truths_scaled.extend(labels_scaled.cpu().numpy().flatten())

            # 修正：正确收集原始尺度的真实值
            truths_original.extend(labels_original_batch)

    valid_loss /= len(valid_dataset)
    valid_losses.append(valid_loss)

    # 将预测值转换回原始浓度尺度
    preds_scaled_np = np.array(preds_scaled)
    preds_original = train_dataset.inverse_transform(preds_scaled_np)
    truths_original = np.array(truths_original)

    # 计算验证集评估指标（在原始尺度上）
    rmse = np.sqrt(mean_squared_error(truths_original, preds_original))
    r2 = r2_score(truths_original, preds_original)
    mae = mean_absolute_error(truths_original, preds_original)
    mape = mean_absolute_percentage_error(truths_original, preds_original)

    valid_rmse_scores.append(rmse)
    valid_r2_scores.append(r2)
    valid_mae_scores.append(mae)
    valid_mape_scores.append(mape)

    # 保存最佳模型
    if rmse < best_rmse:
        best_rmse = rmse
        paddle.save(model.state_dict(), os.path.join(save_dir, 'best_cnn_lstm_model.pdparams'))
        print(f"Epoch {epoch + 1}: 保存最佳模型 (RMSE = {rmse:.4f})")

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid RMSE: {rmse:.4f} | Valid R2: {r2:.4f}")
    print(f"Valid MAE: {mae:.4f} | Valid MAPE: {mape:.4f}")

    # 更新学习率
    lr_scheduler.step()

# 保存最终模型
paddle.save(model.state_dict(), os.path.join(save_dir, 'final_cnn_lstm_model.pdparams'))
paddle.save(optimizer.state_dict(), os.path.join(save_dir, 'final_cnn_lstm_optimizer.pdopt'))

# 绘图
# 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
plt.show()

# 评估指标曲线（RMSE）
plt.figure(figsize=(10, 5))
plt.plot(valid_rmse_scores, label='Valid RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Validation RMSE')
plt.legend()
plt.savefig(os.path.join(save_dir, 'rmse_curve.png'))
plt.show()

# 评估指标曲线（R2）
plt.figure(figsize=(10, 5))
plt.plot(valid_r2_scores, label='Valid R2')
plt.xlabel('Epoch')
plt.ylabel('R2 Score')
plt.title('Validation R2 Score')
plt.legend()
plt.savefig(os.path.join(save_dir, 'r2_curve.png'))
plt.show()

# 评估指标曲线（MAE）
plt.figure(figsize=(10, 5))
plt.plot(valid_mae_scores, label='Valid MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Validation MAE')
plt.legend()
plt.savefig(os.path.join(save_dir, 'mae_curve.png'))
plt.show()

# 预测与真实值对比
plt.figure(figsize=(12, 6))
plt.scatter(truths_original, preds_original, alpha=0.7)
plt.plot([min(truths_original), max(truths_original)], [min(truths_original), max(truths_original)], 'r--')
plt.xlabel('True Concentration')
plt.ylabel('Predicted Concentration')
plt.title('True vs Predicted Concentration')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(save_dir, 'true_vs_predicted.png'))
plt.show()

# 序列预测可视化（随机选择一个样本）
sample_idx = np.random.randint(0, len(valid_dataset))
sample_imgs, sample_label_scaled, sample_label_original = valid_dataset[sample_idx]
sample_imgs = sample_imgs.unsqueeze(0)  # 添加batch维度

model.eval()
with paddle.no_grad():
    sample_pred_scaled = model(sample_imgs).numpy()[0][0]
    # 转换回原始尺度
    sample_pred_original = train_dataset.inverse_transform(np.array([sample_pred_scaled]))[0]
    sample_label_original = float(sample_label_original)

plt.figure(figsize=(15, 5))
plt.plot(range(sample_imgs.shape[1]), [sample_label_original] * sample_imgs.shape[1], 'r-', label='True')
plt.plot(range(sample_imgs.shape[1]), [sample_pred_original] * sample_imgs.shape[1], 'b--', label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Concentration')
plt.title(f'Concentration Prediction for Sample {sample_idx}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(save_dir, 'sample_prediction.png'))
plt.show()    