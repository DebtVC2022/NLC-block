import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# 生成模拟数据
np.random.seed(0)
X = np.random.rand(1000, 1, 28, 28)
y = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])

# 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 修改部分训练集标签为相反的标签
num_samples_to_change_train = int(0.1 * len(y_train))
change_indices_train = np.random.choice(len(y_train), num_samples_to_change_train, replace=False)
y_train_changed = y_train.copy()
y_train_changed[change_indices_train] = 1 - y_train[change_indices_train]

# 修改部分验证集标签为相反的标签
num_samples_to_change_val = int(0.1 * len(y_val))
change_indices_val = np.random.choice(len(y_val), num_samples_to_change_val, replace=False)
y_val_changed = y_val.copy()
y_val_changed[change_indices_val] = 1 - y_val[change_indices_val]

# 修改部分测试集标签为相反的标签
num_samples_to_change_test = int(0.1 * len(y_test))
change_indices_test = np.random.choice(len(y_test), num_samples_to_change_test, replace=False)
y_test_changed = y_test.copy()
y_test_changed[change_indices_test] = 1 - y_test[change_indices_test]

# 转换数据为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
y_train_changed = torch.LongTensor(y_train_changed)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
y_val_changed = torch.LongTensor(y_val_changed)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)
y_test_changed = torch.LongTensor(y_test_changed)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型和定义损失函数与优化器
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
batch_size = 32
epochs = 100
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train_changed[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 每隔5轮验证模型
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs, 1)
            correct_noisy = (predicted == y_val_changed).sum().item()
            accuracy_val_noisy = correct_noisy / len(y_val_changed)
            print(f"Validation accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")

            correct_true = (predicted == y_val).sum().item()
            accuracy_val_true = correct_true / len(y_val)
            print(f"Validation accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")

        model.train()

# 最终测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    correct_noisy = (predicted == y_test_changed).sum().item()
    accuracy_test_noisy = correct_noisy / len(y_test_changed)
    print(f"Final Test accuracy on noisy label: {accuracy_test_noisy}")

    correct_true = (predicted == y_test).sum().item()
    accuracy_test_true = correct_true / len(y_test)
    print(f"Final Test accuracy on true label: {accuracy_test_true}")