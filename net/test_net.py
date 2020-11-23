from pointnet_cls import PointnetCls
from pointnet_part import PointnetPart
import torch
import torch.optim as optim
import torch.nn.functional as F


def test_cls_net(device):
    net = PointnetCls(10).to(device)
    points = torch.rand(3, 3, 128).to(device)
    label = torch.Tensor([1, 2, 3]).long().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    tot_loss = 0
    for i in range(5):
        optimizer.zero_grad()
        pred, _ = net(points)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print("train classifier succeeded. loss %7f"%tot_loss)


def test_part_net(device):
    net = PointnetPart(10).to(device)
    points = torch.rand(3, 3, 128).to(device)
    label = torch.ones(3, 128, 1).long().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    tot_loss = 0
    for i in range(5):
        optimizer.zero_grad()
        pred, _ = net(points)
        loss = F.nll_loss(pred.view(-1, 10), label.view(-1))
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print("train part seg succeeded. loss %7f" % tot_loss)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_cls_net(device)
    test_part_net(device)


if __name__ == "__main__":
    main()