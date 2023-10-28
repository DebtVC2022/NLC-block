import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.datasets as datasets

from random import choice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
batch_size = 32
num_classes = 2
learning_rate = 0.001
num_epochs = 100
noisy_ratio = 0.2

dataset_train = datasets.ImageFolder('./chest_xray/train', transform)
# 对应文件夹的label
print(dataset_train.class_to_idx)
dataset_test = datasets.ImageFolder('./chest_xray/val', transform)
# 对应文件夹的label
print(dataset_test.class_to_idx)
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


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
        
        self.fc = nn.Linear(4096, num_classes)
        
    # 定义前向传播顺序
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    

model = ConvNet(num_classes).to(device)
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
        images = images.to(device)
        labels = labels.to(device)
        labels_np = labels.clone()#.numpy()
        num_samples_to_change_train = int(noisy_ratio * len(labels_np))
        change_indices_train = np.random.choice(len(labels_np), num_samples_to_change_train, replace=False)
        labels_changed = labels_np.clone()

        #labels_changed[change_indices_train] = [(each-1) if each ==9 else (each+1) for each in labels_np[change_indices_train]]
        for each in labels_np[change_indices_train]:
            result_list = list(range(0, 2))
            result_list.remove(each)
            labels_changed[change_indices_train] = choice(result_list)
        #labels_changed = torch.LongTensor(labels_changed)
        
        # 前向传播
        outputs = model(images)
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
            for testimages, testlabels in test_loader:
                testimages = testimages.to(device)
                testlabels = testlabels.to(device)
                
                
                outputs = model(testimages)
                _, predicted = torch.max(outputs.data, 1)
                total += testlabels.size(0)
                correct_true += (predicted == testlabels).sum().item()

                testlabels_np = testlabels.clone()#.numpy()
                num_samples_to_change_testlabels = int(noisy_ratio * len(testlabels_np))
                change_indices_testlabels = np.random.choice(len(testlabels_np), num_samples_to_change_testlabels, replace=False)
                testlabels_changed = testlabels_np.clone()

                #testlabels_changed[change_indices_testlabels] = [(each-1) if each ==9 else (each+1) for each in testlabels_np[change_indices_testlabels]]
                for eachtest in testlabels_np[change_indices_testlabels]:
                    result_list = list(range(0, 2))
                    result_list.remove(eachtest)
                    testlabels_changed[change_indices_testlabels] = choice(result_list)
                #testlabels_changed = torch.LongTensor(testlabels_changed)

                correct_noisy += (predicted == testlabels_changed).sum().item()
            
            accuracy_val_noisy = correct_noisy / total
            accuracy_val_true = correct_true / total
            
            print(f"Validation accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")
            print(f"Validation accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")
        model.train()

dataset_test = datasets.ImageFolder('./chest_xray/test', transform)
# 对应文件夹的label
print(dataset_test.class_to_idx)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
model.eval()
with torch.no_grad():
    correct_true = 0
    correct_noisy = 0
    total = 0
    for testimages, testlabels in test_loader:
        testimages = testimages.to(device)
        testlabels = testlabels.to(device)
        
        outputs = model(testimages)
        _, predicted = torch.max(outputs.data, 1)
        total += testlabels.size(0)
        correct_true += (predicted == testlabels).sum().item()

        testlabels_np = testlabels.clone()#.numpy()
        num_samples_to_change_testlabels = int(noisy_ratio * len(testlabels_np))
        change_indices_testlabels = np.random.choice(len(testlabels_np), num_samples_to_change_testlabels, replace=False)
        testlabels_changed = testlabels_np.clone()

        for eachtest in testlabels_np[change_indices_testlabels]:
            result_list = list(range(0, 2))
            result_list.remove(eachtest)
            testlabels_changed[change_indices_testlabels] = choice(result_list)

        correct_noisy += (predicted == testlabels_changed).sum().item()
    
    accuracy_val_noisy = correct_noisy / total
    accuracy_val_true = correct_true / total
    
    print(f"Testing accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")
    print(f"Testing accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")