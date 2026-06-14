import torch
from d2l import torch as d2l
from torch import nn
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return(H@W2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1

updater = torch.optim.SGD(params, lr=lr)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在 n 个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期, 返回 (训练损失, 训练精度)"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 训练损失总和、训练正确数、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并绘制训练曲线"""
    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'epoch {epoch + 1}, '
              f'train loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}, '
              f'test acc {test_acc:.4f}')

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs, label='train acc', linestyle='--')
    plt.plot(epochs, test_accs, label='test acc', linestyle='-.')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('MLP (从零实现) 训练效果')
    plt.legend()
    plt.grid(True)
    plt.savefig('mlp_from_zero_train.png')
    plt.show()


if __name__ == '__main__':
    train(net, train_iter, test_iter, loss, num_epochs, updater)
