'''https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch'''


import torch.nn as nn
import torch.nn.functional as F

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out