import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
            nn.ReLU(), # 激活函数，增加非线性

            # 第二层：128->64
            nn.Linear(128, 64),
            nn.ReLU(),

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
for epoch in range(1, 6):
    model.train()
    total_loss = 0

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
    print(f'第{epoch}轮：平均误差（Loss）：{avg_loss: .3f}， 准确率：{accuracy: .2f}%')