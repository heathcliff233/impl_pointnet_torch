from torchsummary import summary
from pointnet_cls import PointnetCls

clsnet = PointnetCls(10)
print("pointnet for classification")
summary(clsnet, (3, 100))
