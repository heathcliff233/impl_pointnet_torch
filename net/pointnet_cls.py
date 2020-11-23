import torch.nn as nn
import torch.nn.functional as F
from pointnet import PointFeature


class PointnetCls(nn.Module):
    def __init__(self, num_classes=10):
        super(PointnetCls, self).__init__() 
        self.feat = PointFeature()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x, trans, transfeat = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans

