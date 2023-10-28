import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

from random import choice


# 数据增广方法
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(), 
    # 随机裁剪至32x32
    transforms.RandomCrop(32), 
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
#     transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))
    ])

# cifar10路径
cifar10Path = './cifar'
batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 100

#  训练数据集
train_dataset = torchvision.datasets.CIFAR10(root=cifar10Path, train=True, transform=transform, download=True)

# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root=cifar10Path,train=False, transform=transform)

# 生成数据加载器
# 训练数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
# 测试数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)

class NoisyLabelCorrectionLayer(nn.Module):
    def __init__(self):
        super(NoisyLabelCorrectionLayer, self).__init__()

    def forward(self, y_hat_i, y_i):
        # y_pred_i = torch.max(y_hat_i, 1).indices
        exponent = y_hat_i
        denominator = 1 + y_i * y_hat_i  # Corrected the denominator
        noisy_correction = exponent / denominator
        return noisy_correction.reshape(y_hat_i.shape[0], -1)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层计算
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            #  批归一化
            nn.BatchNorm2d(32),
            #ReLU激活函数
            nn.ReLU(),
            # 池化层：最大池化
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc0 = nn.Linear(4096, num_classes)
        #self.fc = nn.Linear(num_classes, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.noisy_correction = NoisyLabelCorrectionLayer()
        
    # 定义前向传播顺序
    def forward(self, x, y):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.softmax(out)
        out = self.noisy_correction(out, y)
        #out = self.fc(out)
        #print(out.shape)
        return out
    

model = ConvNet(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

torch.manual_seed(3407)
np.random.seed(10)
random.seed(20)
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 对每个batch中的标签进行修改
        labels_np = labels.clone().numpy()
        num_samples_to_change_train = int(0.2 * len(labels_np))
        change_indices_train = np.random.choice(len(labels_np), num_samples_to_change_train, replace=False)
        labels_changed = labels_np.copy()

        #labels_changed[change_indices_train] = [(each-1) if each ==9 else (each+1) for each in labels_np[change_indices_train]]
        for each in labels_np[change_indices_train]:
            result_list = list(range(0, 10))
            result_list.remove(each)
            labels_changed[change_indices_train] = choice(result_list)
        labels_changed = torch.LongTensor(labels_changed)

        labels_changed_onehot = torch.zeros(labels_changed.shape[0], num_classes)
        for i in range(labels_changed.shape[0]):
            max_index = labels_changed[i]
            labels_changed_onehot[i, max_index] = 1
        
        # 前向传播
        outputs = model(images, labels_changed_onehot)
        loss = criterion(outputs, labels_changed)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            correct_true = 0
            correct_noisy = 0
            total = 0
            for images, testlabels in test_loader:
                testlabels_np = testlabels.clone().numpy()
                num_samples_to_change_testlabels = int(0.2 * len(testlabels_np))
                change_indices_testlabels = np.random.choice(len(testlabels_np), num_samples_to_change_testlabels, replace=False)
                testlabels_changed = testlabels_np.copy()

                #testlabels_changed[change_indices_testlabels] = [(each-1) if each ==9 else (each+1) for each in testlabels_np[change_indices_testlabels]]
                for eachtest in testlabels_np[change_indices_testlabels]:
                    result_list = list(range(0, 10))
                    result_list.remove(eachtest)
                    testlabels_changed[change_indices_testlabels] = choice(result_list)
                testlabels_changed = torch.LongTensor(testlabels_changed)


                testlabels_changed_onehot = torch.zeros(testlabels_changed.shape[0], num_classes)
                for i in range(testlabels_changed.shape[0]):
                    testlabels_max_index = testlabels_changed[i]
                    testlabels_changed_onehot[i, testlabels_max_index-1] = 1

                outputs = model(images, testlabels_changed_onehot)
                _, predicted = torch.max(outputs.data, 1)
                total += testlabels.size(0)
                correct_true += (predicted == testlabels).sum().item()

                correct_noisy += (predicted == testlabels_changed).sum().item()
            
            accuracy_val_noisy = correct_noisy / total
            accuracy_val_true = correct_true / total
            
            print(f"Validation accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")
            print(f"Validation accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")
        model.train()
