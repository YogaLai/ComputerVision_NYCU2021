import torch
from util import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
from dataloader import ImgDataset
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp')
args = parser.parse_args()
if args.exp != None:
    log = 'logs/' + args.exp
else:
    log = 'logs/'

def train(model, train_loader, criterion, optimizer):
    correct = 0
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        input, label = data[0], data[1]
        input, label = input.cuda(), label.cuda()
        output = model(input)
        _, predict = torch.max(output.data, 1)
        correct+=(predict==label).sum().item()
        
        loss = criterion(output, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return total_loss/len(train_loader.dataset), correct/len(train_loader.dataset)*100

def evaluate(model, test_loader):
    with torch.no_grad():
        correct = 0
        for data in test_loader:
            input, label = data[0], data[1]
            input, label = input.cuda(), label.cuda()
            output = model(input)
            _, predict = torch.max(output.data, 1)
            correct+=(predict==label).sum().item()
    
    return correct/len(test_loader.dataset)*100

batch_size = 64
lr = 0.001
num_classes = 15
epochs = 50
trainset = ImgDataset(train_mode=True)
testset = ImgDataset(train_mode=False)
train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(testset,batch_size=batch_size,shuffle=True)

model = models.resnet18()
num_neurons = model.fc.in_features
model.fc=nn.Linear(num_neurons, num_classes)
model=model.cuda()
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(log)
for epoch in range(epochs):
    loss, acc = train(model, train_loader, criterion, optimizer)
    print('Epoch %d: loss=%f, acc=%f' % (epoch, loss, acc))
    writer.add_scalar('Train/loss', loss, epoch)
    writer.add_scalar('Train/acc', acc, epoch)
    
    acc = evaluate(model, test_loader)
    writer.add_scalar('Test/acc', acc, epoch)
