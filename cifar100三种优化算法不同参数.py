import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os

# 超参数
epochs = 50
batch_size = 128
learning_rate = 0.1  # 固定学习率
weight_decays = [5e-4, 1e-4, 5e-5, 1e-5]  # 四种权重衰减
rho = 0.05  # SAM的扰动范围
train_ratio = 1  # 训练集比例（10%）

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 的均值和标准差
])

# 加载CIFAR-100数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 按比例划分训练集
train_size = int(train_ratio * len(train_dataset))
train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 定义ResNet18的基本模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义ResNet18
class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=100):  # CIFAR-100 有 100 个类别
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 初始化模型
def init_model():
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=100)
    model = model.to(device)
    return model

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义SAM优化器
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group['rho'] = rho  # 确保每个参数组都有 'rho'

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual optimization step
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# 训练和测试函数
def train(epoch, model, optimizer, scheduler, is_sam=False):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if is_sam:
            optimizer.first_step(zero_grad=True)
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss = train_loss / len(train_loader)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch: {epoch} | Train Loss: {loss:.3f} | Train Acc: {acc:.3f}% | Time: {epoch_time:.2f}s')
    return loss, acc

def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    loss = test_loss / len(test_loader)
    print(f'Epoch: {epoch} | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}%')
    return loss, acc

# 保存结果
def save_results(config_name, train_losses, train_accuracies, test_losses, test_accuracies, train_time, final_train_acc, final_test_acc):
    output_dir = os.path.join('output1', config_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'loss_accuracy.png'))
    plt.close()

    # 保存结果到文本文件
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f'Training Time: {train_time:.2f}s\n')
        f.write(f'Final Train Accuracy: {final_train_acc:.3f}%\n')
        f.write(f'Final Test Accuracy: {final_test_acc:.3f}%\n')


# 主函数
def main():
    optimizers = {
        # SGD优化器，学习率设置为0.01
        'SGD': lambda model, weight_decay: optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay),
        # Adam优化器，学习率设置为0.001
        'Adam': lambda model, weight_decay: optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay),
        # SAM优化器，基于SGD，学习率设置为0.01
        'SAM': lambda model, weight_decay: SAM(model.parameters(), optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay), rho=rho)
    }

    for opt_name, opt_fn in optimizers.items():
        for weight_decay in weight_decays:
            model = init_model()
            optimizer = opt_fn(model, weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            train_losses, train_accuracies = [], []
            test_losses, test_accuracies = [], []
            start_time = time.time()

            for epoch in range(1, epochs + 1):
                train_loss, train_acc = train(epoch, model, optimizer, scheduler, is_sam=(opt_name == 'SAM'))
                test_loss, test_acc = test(epoch, model)
                scheduler.step()

                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)

            train_time = time.time() - start_time
            final_train_acc = train_accuracies[-1]
            final_test_acc = test_accuracies[-1]

            config_name = f'{opt_name}_weight_decay{weight_decay}'
            save_results(config_name, train_losses, train_accuracies, test_losses, test_accuracies, train_time, final_train_acc, final_test_acc)

if __name__ == '__main__':
    main()