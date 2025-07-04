import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import os

from networks import *

# --------- Fixed Parameters ---------
DATASET = 'cifar100'
DEPTH = 28
WIDEN_FACTOR = 10
DROPOUT_RATE = 0.3
LR = 0.1
BATCH_SIZE = 128
EPOCHS = 100

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DATASET == 'cifar100':
        num_classes = 100
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2761)
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
    else:
        raise ValueError('Only cifar100 is supported in this version.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    net = Wide_ResNet(depth=DEPTH, widen_factor=WIDEN_FACTOR, dropout_rate=DROPOUT_RATE, num_classes=num_classes)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        avg_loss = train_loss / total
        avg_acc = 100. * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        print('Epoch: %d | Loss: %.3f | Acc: %.3f%%' % (epoch, avg_loss, avg_acc))

    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        avg_loss = test_loss / total
        avg_acc = 100. * correct / total
        test_losses.append(avg_loss)
        test_accuracies.append(avg_acc)
        print('Test Loss: %.3f | Acc: %.3f%%' % (avg_loss, avg_acc))

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        if epoch % 10 == 0 or epoch == 1:
            test()
        scheduler.step()

    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) * 10 + 1, 10), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig('results/loss_convergence.png')

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies) * 10 + 1, 10), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.savefig('results/accuracy_convergence.png')

if __name__ == '__main__':
    main()