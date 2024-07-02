import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):
    """ Computes the top 1 accuracy """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_mnist_data(batch_size=64, num_workers=0):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=True)

    return train_loader, test_loader


def get_cifar10_data(batch_size=8, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)

    return train_loader, test_loader


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device, epochs=5):
    s = time.time()
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = AverageMeter('loss')
        acc = AverageMeter('train_acc')
        for batch_idx, (inputs, targets) in enumerate(dataloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.update(loss.item(), outputs.shape[0])
            acc.update(accuracy(outputs, targets), outputs.shape[0])
            if batch_idx % 100 == 0:  # print every 100 mini-batches
                print('[%d, %5d] ' %
                      (epoch + 1, batch_idx + 1), running_loss, acc)
    print('Finished Training')
    elapsed = time.time() - s
    return elapsed


def test(model: nn.Module, testloader: DataLoader, device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total


def distillation_train(student_model, teacher_model, dataloader, device, criterion, optimizer, epochs=5):
    s = time.time()
    student_model.train()
    teacher_model.eval()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = AverageMeter('loss')
        acc = AverageMeter('train_acc')
        for batch_idx, (inputs, targets) in enumerate(dataloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            teacher_outputs = teacher_model(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = student_model(inputs)
            loss = criterion(outputs, teacher_outputs, targets)
            assert not torch.isnan(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.update(loss.item(), outputs.shape[0])
            acc.update(accuracy(outputs, targets), outputs.shape[0])
            if batch_idx % 100 == 0:  # print every 100 mini-batches
                print('[%d, %5d] ' %
                      (epoch + 1, batch_idx + 1), running_loss, acc)
    print('Finished Training')
    elapsed = time.time() - s
    return elapsed


def time_model_evaluation(model, test_data, device):
    s = time.time()
    acc = test(model, test_data, device)
    elapsed = time.time() - s
    return acc, elapsed
    # print('''acc: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(acc, elapsed))


def make_criterion(alpha=0.5, T=4.0, mode='cse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            # 交叉熵
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            # 均方误差
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        elif mode == 'kl':
            # kl散度
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = F.kl_div(_p, _q, reduction='batchmean')
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion
