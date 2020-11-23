import torch
from torchsummary import summary
from pointnet_cls import PointnetCls
from pointnet_part import PointnetPart
'''
clsnet = PointnetCls(10)
print("pointnet for classification")
summary(clsnet, (3, 100))

semnet = PointnetSem(10)
print("pointnet for semantic segmentation")
summary(semnet, (3, 100))
'''

model = PointnetPart(10)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
print("optimizer's state_dict:")
# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
