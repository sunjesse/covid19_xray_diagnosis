import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from sklearn.datasets import load_breast_cancer
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import numpy as np
import torchvision.transforms as transforms
from utils import GaussianBlur

data_augment = transforms.Compose([transforms.ToPILImage(),
                                   transforms.RandomHorizontalFlip(),
                                   GaussianBlur(),
                                   transforms.ToTensor()])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = torchvision.models.densenet121(pretrained=False)
        self.encoder = nn.Sequential(*list(self.base.children())[:-1])
        self.linear = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 4))

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.encoder(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def dataloader(args):
    img_train = np.load("/home/rexma/Desktop/MRI_Images/COVID19/x_train.npy")
    label_train = np.load("/home/rexma/Desktop/MRI_Images/COVID19/y_train.npy")
    img_test= np.load("/home/rexma/Desktop/MRI_Images/COVID19/x_test.npy")
    label_test = np.load("/home/rexma/Desktop/MRI_Images/COVID19/y_test.npy")
    
    img_train, img_test = img_train.transpose(0, 3, 1, 2), img_test.transpose(0, 3, 1, 2)
    
    trainloader = DataLoader(TensorDataset(torch.from_numpy(img_train),
                             torch.from_numpy(label_train).long()),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    testloader = DataLoader(TensorDataset(torch.from_numpy(img_test),
                            torch.from_numpy(label_test).long()),
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last= True)

    return trainloader, testloader

def optimizer(net, args):
    assert args.optimizer.lower() in ["sgd", "adam"], "Invalid Optimizer"

    if args.optimizer.lower() == "sgd":
        return optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta1, nesterov=args.nesterov)
    elif args.optimizer.lower() == "adam":
        return optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

def test(net, criterion, testloader, args):
    net.eval()
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            pred = F.softmax(outputs, 1)
            _, pred = torch.max(pred, 1)
            correct += torch.sum(pred==labels)
        print("Test accuracy: " + str(float(correct)/float(testloader.__len__() * args.batch_size)))

def train(net, criterion, optimizer, trainloader, epoch, args):
    running_loss = 0.0
    correct = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        for idx, x in enumerate(inputs):
            inputs[idx] = data_augment(x)
        
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        pred = F.softmax(outputs, 1)
        _, pred = torch.max(pred, 1)
        correct += torch.sum(pred==labels)
        total += pred.shape[0]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0 and i > 0:
            print('[Epoch %02d, Minibatch %05d] Loss: %.5f, Accuracy: %.2f' %
                        (epoch, i, running_loss/args.batch_size, correct/total))
            running_loss, correct, total = 0.0, 0.0, 0

def checkpoint(net, args, epoch_num):
    print('Saving checkpoints...')

    suffix_latest = 'epoch_{}.pth'.format(epoch_num)
    dict_net = net.state_dict()
    torch.save(dict_net,
               '{}/resnet_{}'.format(args.ckpt, suffix_latest))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--nesterov', default=False)
    parser.add_argument('--ckpt', default="/home/rexma/Desktop/JesseSun/covid19/ckpt")

    args, unknown = parser.parse_known_args()

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    trainloader, testloader = dataloader(args)
    net = torchvision.models.densenet121(pretrained=False, num_classes=4).cuda() 
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optimizer(net, args)

    for epoch in range(1, args.epoch+1):
        train(net, criterion, optimizer, trainloader, epoch, args)
        test(net, criterion, testloader, args)
        if epoch % 25 == 0:
            checkpoint(net, args, epoch)
    print("Training completed")
