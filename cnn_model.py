import torch.nn as nn
import torch.nn.functional as f
import torch


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(115200, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 30)

    def forward(self, x):
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv1(x)), 2), 0.5)
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv2(x)), 2), 0.5)
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv3(x)), 2), 0.5)
        x = x.view(-1, self.num_flat_features(x))
        x = f.dropout(f.relu(self.fc1(x)), 0.5)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    acc = 0.0
    avg_loss = 0.0
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['labels']
        data, target = data.float().to(device), target.float().to(device)
        # print(target[0])
        labels = torch.max(target, 1)[1]
        optimizer.zero_grad()
        output = model(data)
        # print(output.size())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels.long())
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        acc += accuracy(output, labels)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss/10, acc/10))
            acc = 0.0
            avg_loss = 0.0


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0.0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            data, target = sample['image'], sample['labels']
            data, target = data.float().to(device), target.float().to(device)
            labels = torch.max(target, 1)[1]
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels.long())
            test_loss += loss.item()
            acc += accuracy(output, labels)
            print(acc.item()/(i+1))
    test_loss /= 600
    acc = (acc*100)/(i+1)
    print('\nTest set: Average loss: {:.4f}, Accuracy:({}%)\n'.format(
        test_loss, acc))


def accuracy(out, gt):
    # print("Out size: ", f.softmax(out).size())
    pred = torch.max(f.softmax(out), 1)[1]
    # print("Pred: ", pred)
    # print("GT:", gt)
    # print((pred == gt).sum()/32)
    return torch.mean((pred == gt).float())

