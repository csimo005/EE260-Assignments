import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self._fc = nn.Linear(28*28, 10, bias=False)
        self._fc.weight.data.zero_()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self._fc(x)
        return x

def create_one_hot(labels):
    one_hot = torch.zeros(labels.shape[0], 10)
    for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1

    return one_hot

def train(model, optimizer, criterion, trainloader, epoch):
    for i, data in enumerate(trainloader):
        img, label = data
        label = create_one_hot(label)

        optimizer.zero_grad()
        scores = model(img)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        output_str = 'Training[%d, %d] total loss: %.3f' % (epoch, i*100, loss.item())
        print(output_str)

def test(model, criterion, testloader, epoch):
    test_loss = 0.
    top_1 = 0.
    total = 0.
    forwards = 0.
    for i, data in enumerate(testloader):
        img, label = data
        label = create_one_hot(label)

        scores = model(img)
        loss = criterion(scores, label)

        test_loss += loss.item()
        total += img.shape[0]
        forwards += 1

        top_1 += torch.sum(torch.max(scores,1)[1] == torch.max(label,1)[1])

    output_str = 'Testing[%d] test loss: %.3f, accuracy: %.3f' % (epoch, test_loss/forwards, 100*(top_1/total))
    print(output_str)

def main(batchsize, lr, epochs):
    tform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST('./', train=True, transform=tform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)

    testset = torchvision.datasets.MNIST('./', train=False, transform=tform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True)

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        train(model, optimizer, criterion, trainloader, epoch)
        test(model, criterion, testloader, epoch)

if __name__ == '__main__':
    main(100, 0.001, 10)
