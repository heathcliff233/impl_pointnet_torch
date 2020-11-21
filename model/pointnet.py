import torch
import torch.nn as nn
import torch.nn.functional as F

class Transform(nn.Module):
    def __init__(self, k):
        super(Transform, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(  k,  64, 1)
        self.conv2 = nn.Conv1d( 64, 128, 1)
        self.conv3 = nn.Conv1d(128,1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear( 512, 256)
        self.fc3 = nn.Linear( 256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_sz, _, num_points = x.size()
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(num_points)(x)
        #x, _ = torch.max(x, dim=2, keepdim=True)  
        x = x.view(batch_sz, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, self.k, self.k)
        res = res.transpose(1, 2)
        x = torch.bmm(res, x)
        x = x + res                        # add identity mateix to enhance the learning of transform matrix 
        x = x.transpose(1, 2)

        return x
        


class PointFeature(nn.Module):
    def __init__(self, global_feature=True, feature_transform=True):
        super(PointFeature, self).__init__()
        self.input_trans = Transform(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_trans = Transform(k=64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_trans(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.feature_trans(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2, keepdims=True)
        x = x.view(-1, 1024)

        return x






