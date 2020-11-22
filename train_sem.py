import argparse
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from model.pointnet_sem import PointnetSem

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size (default to 24')
    parser.add_argument('--trained_model', default='', help='pre-trained model path (default to none)')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training (default to 200)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training (default to 0.01)')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate (default to 0.1)')
    parser.add_argument('--regulize', type=bool, default=True, help='to regulize with tranform matrix')
    return parser.parse_args()

def train(model, loader, optim, regulize=True, epoch, opt, device, classes):
    model.train()
    correct, train_loss = 0
    for i, data in enumerate(loader):
        points, label = data
        #points = points.transpose(1, 2)
        points, label = points.to(device), label.to(device)
        optim.zero_grad()
        pred, trans = model(points)
        pred = pred.view(-1, classes)
        label = label.view(-1)
        loss = F.nll_loss(pred, label)
        I = torch.eye(3).view(1,3,3)
        I = trans.is_cuda() ? I.to(torch.device("cuda:0")) : I
        loss += torch.mean(torch.norm(torch.bmm(trans, trans.transpose(1,2))-I, dim=(1,2)))*0.001
        loss.backward()
        optim.step()
        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss += loss.item()
        print("[epoch %3d: batch %3d] train loss: %7f accuracy %7f" % (epoch, i, train_loss, correct/float(opt.batch_size)))


def test(model, loader, epoch, opt, device, classes):
    model.eval()
    correct, train_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            points, label = data
            #points = points.transpose(1, 2)
            points, label = points.to(device), label.to(device)
            optim.zero_grad()
            pred, trans = model(points)
            pred = pred.view(-1, classes)
            label = label.view(-1)
            loss = F.nll_loss(pred, label)
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            train_loss += loss.item()
    print("[epoch %3d: batch %3d] test loss: %7f accuracy %7f" % (epoch, i, train_loss, correct/float(len(loader.dataset))))

def main(opt):
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    num_classes = len(train_set.classes)
    model = PointnetSem(num_classes).to(device)
    if opt.model != "" :
        model.load_state_dict(opt.model)
    optim = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    scheduler = optim.StepLR(optim, step_size=20, gamma=opt.decay_rate)

    for epoch in range(1, opt.epoch + 1):
        scheduler.step()
        train(model, loader, optim, regulize=True, epoch=epoch, opt=opt, device, num_classes)
        test(model, loader, epoch, opt=opt, device, num_classes)

if __name__ == "__main__":
    opt = parse_args()
    main(opt)