import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 三通道的均值方差
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 定义模型(基线模型)
class My_CNN(nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()

        # 卷积层（特征提取）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # 池化：把图片缩小一半

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 再池化一次

        # 全连接层（分类）
        self.flatten = nn.Flatten()

        # 图片经过conv2，通道变为64
        # 两次pool（缩小两倍），图片尺寸32->16->8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 10种类型

    def forward(self, x):
        # 第一层卷积：卷积->激活->池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第二层卷积
        x = self.pool(self.relu(self.conv2(x)))

        # 展平
        x = self.flatten(x)

        # 全连接分类
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = My_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) #动量加速

loss_history = []
acc_history = []

epochs = 20

for epoch in range(1, epochs+1):
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    # 测试
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        acc_history.append(acc)
        print(f"第{epoch}轮：平均误差（Loss）：{avg_loss: .3f}, 准确率：{acc: .2f}%")

# 画图
plt.figure(figsize=(12, 5))

# 图1: 损失 (Loss)
plt.subplot(1, 2, 1)

plt.plot(range(1, epochs + 1), loss_history, 'b-o', label='Training Loss')
plt.title('CNN Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7) # 加网格，透明度0.7
plt.legend()

# 图2: 准确率 (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), acc_history, 'r-s', label='Test Accuracy')
plt.title(f'CNN Test Accuracy (Best: {max(acc_history):.2f}%)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()