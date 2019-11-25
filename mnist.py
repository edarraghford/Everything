"""
Classification of handwritten digits
====================================

Based on pytorch example for MNIST
"""


import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import kymatio.datasets as scattering_datasets
import kymatio
import torch
import argparse
import math
import data 
import numpy as np 

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(28224, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3   = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out) 
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.sigmoid(self.fc3(out))
        return out

def train(model, device, train_loader, optimizer, epoch, scattering, prnt=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target= torch.autograd.Variable(data), torch.autograd.Variable(target)        

        optimizer.zero_grad()
        output = model(scattering(data))
        if(prnt == True): 
            print(target)
            print(output) 
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, scattering):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.binary_cross_entropy(output, target) # , reduction='sum').item() # sum up batch loss
            out = np.array(output)
            targ = np.array(target)
            out[out >= 0.5] = 1 
            out[out < 0.5] = 0 
            pred = 1- np.abs(np.transpose(out)-targ) # get the index of the max log-probability
            print(targ)
            print(output)
            correct += np.sum(pred)
    test_loss /= len(test_loader.dataset)
    print(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """Train a simple Hybrid Scattering + CNN model on MNIST.

        Three models are demoed:
        'linear' - scattering + linear model
        'mlp' - scattering + MLP
        'cnn' - scattering + CNN

        scattering 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.

        scatter + linear achieves 99.15% in 15 epochs
        scatter + cnn achieves 99.3% in 15 epochs

    """
    parser = argparse.ArgumentParser(description='MNIST scattering  + hybrid examples')
    parser.add_argument('--mode', type=int, default=2,help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='cnn',help='classifier model')
    args = parser.parse_args()
#    assert(args.classifier in ['linear','mlp','cnn'])
#    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.mode == 1:
        scattering = Scattering2D(J=2, shape=(180, 180), max_order=1)
        K = 17
    else:
        scattering = Scattering2D(J=2, shape=(180, 180))
        K = 81
    if use_cuda:
        scattering = scattering.cuda()


    

    if args.classifier == 'cnn':
        model = nn.Sequential(
            View(K, 45, 45),
            nn.BatchNorm2d(K),
            nn.Conv2d(K, 64, 3, padding=1), nn.ReLU(),
            View(64*45*45),
            nn.Linear(64* 45 * 45, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        ).to(device)

    elif args.classifier == 'mlp':
        model = nn.Sequential(
            View(K, 45, 45),
            nn.BatchNorm2d(K),
            View(K*45*45),
            nn.Linear(K*45*45, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    elif args.classifier == 'linear':
        model = nn.Sequential(
            View(K, 45, 45),
            nn.BatchNorm2d(K),
            View(K * 45 * 45),
            nn.Linear(K *45 *45, 1), nn.Sigmoid()
        )
    else:
        model = LeNet()
    model.to(device)
    print(model) 
    #initialize
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            m.weight.data.normal_(0, 2./math.sqrt(n))
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 2./math.sqrt(m.in_features))
            m.bias.data.zero_()

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False
    train_loader = torch.utils.data.DataLoader(data.SimulationsDataset(image_file='train_obs.txt', target_file = 'train_labels2.txt'), batch_size=23, shuffle=True)

#    train_loader = torch.utils.data.DataLoader(
#        datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=True, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ])),
#        batch_size=128, shuffle=True) #, num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(data.SimulationsDataset(image_file='valid_obs.txt', target_file = 'valid_labels2.txt'), batch_size=25, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data.SimulationsDataset(image_file='test_obs.txt', target_file = 'test_labels2.txt'), batch_size=53, shuffle=True)

#    test_loader = torch.utils.data.DataLoader(
#        datasets.MNIST(scattering_datasets.get_dataset_dir('MNIST'), train=False, transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,))
#        ])),
#        batch_size=128, shuffle=True) #, num_workers=num_workers, pin_memory=pin_memory)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001 , momentum = 0.9, weight_decay=0.0005)

    for epoch in range(1, 9):
        train( model, device, train_loader, optimizer, epoch, scattering)
        test(model, device, valid_loader, scattering)
#
#        if (epoch%4== 0):
#            train( model, device, train_loader, optimizer, epoch, scattering, prnt=True)
#            test(model, device, valid_loader, scattering)
    test(model, device, test_loader, scattering)

 #       else: 
 #           train( model, device, train_loader, optimizer, epoch, scattering, prnt=False)

if __name__ == '__main__':
    main()
