import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
import torchvision.datasets as datasets
import torchvision.models as models
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

batch_size = 32
num_classes = 2
learning_rate = 0.001
num_epochs = 300
noisy_ratio_lst = [0.1, 0.2, 0.3]

for noisy_ratio in noisy_ratio_lst:
    dataset_train = datasets.ImageFolder('./chest_xray/train', transform)
    # 对应文件夹的label
    print(dataset_train.class_to_idx)
    dataset_test = datasets.ImageFolder('./chest_xray/val', transform)
    # 对应文件夹的label
    print(dataset_test.class_to_idx)
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    class NoisyLabelCorrectionLayer(nn.Module):
        def __init__(self):
            super(NoisyLabelCorrectionLayer, self).__init__()

        def forward(self, y_hat_i, y_i):
            exponent = y_hat_i * y_hat_i
            denominator = 1 + y_i * y_hat_i  # Corrected the denominator
            noisy_correction = exponent / denominator
            return noisy_correction.reshape(y_hat_i.shape[0], -1)
        
    class CustomModel(nn.Module):
        def __init__(self, num_classes=4):
            super(CustomModel, self).__init__()
            # 加载预训练的 ResNet34
            self.base_model = models.resnet34(weights='IMAGENET1K_V1')
            # 替换最后一层全连接层，输出特征向量
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, 512)
            self.noisy_correction = NoisyLabelCorrectionLayer()
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x, y):
            features = self.base_model(x)  # 提取特征
            corrected_features = self.noisy_correction(features, y.reshape(y.shape[0], 1))
            out = self.fc(corrected_features)  # 分类输出
            return out
        

    model = CustomModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    torch.manual_seed(3407)
    np.random.seed(10)
    random.seed(20)
    total_step = len(train_loader)

    loss_all = []
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # 对每个batch中的标签进行修改
            images = images.to(device)
            labels = labels.to(device)
            labels_np = labels.clone() #.numpy()
            num_samples_to_change_train = int(noisy_ratio * len(labels_np))
            change_indices_train = np.random.choice(len(labels_np), num_samples_to_change_train, replace=False)
            labels_changed = labels_np.clone()

            # 对称噪声和非对称噪声一致
            candidates = {
                0: [1],
                1: [0]
            }
            
            random_labels = torch.tensor(
                [choice(candidates[label.item()]) for label in labels_np[change_indices_train]], 
                device=labels_np.device
            )
            labels_changed[change_indices_train] = random_labels.long()
            
            # 前向传播
            outputs = model(images, labels_changed)
            loss = criterion(outputs, labels_changed)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 2 == 0:
            model.eval()
            loss_all.append(loss)
            with torch.no_grad():
                correct_true = 0
                correct_noisy = 0
                total = 0
                for testimages, testlabels in test_loader:
                    testimages = testimages.to(device)
                    testlabels = testlabels.to(device)
                    testlabels_np = testlabels.clone()#.numpy()
                    num_samples_to_change_testlabels = int(noisy_ratio * len(testlabels_np))
                    change_indices_testlabels = np.random.choice(len(testlabels_np), num_samples_to_change_testlabels, replace=False)
                    testlabels_changed = testlabels_np.clone()

                    # 对称噪声和非对称噪声一致
                    test_candidates = {
                        0: [1],
                        1: [0]
                    }
                    
                    random_labels = torch.tensor(
                        [choice(test_candidates[label.item()]) for label in testlabels_np[change_indices_testlabels]], 
                        device=testlabels_np.device
                    )
                    testlabels_changed[change_indices_testlabels] = random_labels.long()

                    outputs = model(testimages, testlabels_changed)
                    _, predicted = torch.max(outputs.data, 1)
                    total += testlabels.size(0)
                    correct_true += (predicted == testlabels).sum().item()
                    correct_noisy += (predicted == testlabels_changed).sum().item()
                
                accuracy_val_noisy = correct_noisy / total
                accuracy_val_true = correct_true / total
                
                print(f"Validation accuracy on noisy after {epoch + 1} epochs: {accuracy_val_noisy}")
                print(f"Validation accuracy on true after {epoch + 1} epochs: {accuracy_val_true}")
            model.train()
            

    pd.DataFrame(loss_all).to_csv("nlc_layer_chest_xray_" + str(noisy_ratio) + ".csv")
