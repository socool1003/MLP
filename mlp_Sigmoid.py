import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义模型
class My_MLP(nn.Module):
    def __init__(self):
        super(My_MLP, self).__init__()

        # 图片转换为向量
        self.flatten = nn.Flatten()

        # 定义层级结构
        self.layers = nn.Sequential(
            # 第一层：784像素->128特征
            nn.Linear(784, 128),
            nn.Sigmoid(), # 激活函数，增加非线性

            # 第二层：128->64
            nn.Linear(128, 64),
            nn.Sigmoid(),

            # 输出层：64->10
            nn.Linear(64, 10),
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

# 实例化模型
model = My_MLP()
print(model)

# 数据集下载准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # 来源于网络，经验数据
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义 损失函数 和 优化器
criterion = nn.CrossEntropyLoss()
# SGD：随机梯度下降  学习率：lr=0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练
train_losses = []  # 记录每一轮的训练损失
test_accuracies = [] # 记录每一轮的测试准确率

epochs = 20

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X)
        # 计算误差
        loss = criterion(predictions, batch_y)
        # 反向传播（求导）
        optimizer.zero_grad()
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    model.eval() # 开始测试
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            predicted_label = predictions.argmax(dim=1)
            correct += (predicted_label == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'第{epoch}轮：平均误差（Loss）：{avg_loss: .3f}， 准确率：{accuracy: .2f}%')

# 作图
plt.figure(figsize = (12, 5))

# 图1: 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-o', label='Training Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# 图2: 测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), test_accuracies, 'r-s', label='Test Accuracy')
plt.title(f"Test Accuracy (Best: {max(test_accuracies):.2f}%)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.show()
