import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet import PointFeature

class PointnetSem(nn.Module):
    def __init__(self, num_classes=10):
        super(PointnetSem, self).__init__()
        self.feat = PointFeature(global_feature=True, feature_transform=True)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        B, C, N = x.size()
        x, transfeat = self.feat(x)
        x = x.view(-1, 1024, 1).repeat(1,1,N)
        x = torch.cat([x, transfeat], 1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(1, 2)
        x = F.log_softmax(x, dim=-1)

        return x

