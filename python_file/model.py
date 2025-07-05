import torch.nn as nn
import torch
from pointnet_utils import PointNetEncoder

class image2sequence(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(image2sequence, self).__init__()
        #input : 1*16*16
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #88
            
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), #44

            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) #22
        )

        self.fc = nn.Sequential(
            nn.Linear(64*2*2, 256),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

class sequence2sequence(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.5):
        super(sequence2sequence, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        neuron_num = [input_dim] + hidden_dim + [output_dim]
        layers = []
        for i in range(1, len(neuron_num)):
            layers.append(nn.Linear(neuron_num[i-1], neuron_num[i]))
            if i < len(neuron_num) - 1:
                layers.append(nn.BatchNorm1d(neuron_num[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

class MLPS(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.5):
        super(MLPS, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.num_phi_slices = output_dim[1]
        self.theta_dim = output_dim[0]

        self.mlps = nn.ModuleList()
        neuron_num = [input_dim] + hidden_dim + [self.theta_dim]

        for _ in range(self.num_phi_slices):
            layers = []
            for j in range(1, len(neuron_num)):
                layers.append(nn.Linear(neuron_num[j-1], neuron_num[j]))
                if j < len(neuron_num) - 1:
                    layers.append(nn.BatchNorm1d(neuron_num[j]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.layers = nn.Sequential(*layers)
            self.mlps.append(self.layers)
        

    def forward(self, x):
        output = list()
        for i in range(self.num_phi_slices):
            output.append(self.mlps[i](x))
        x = torch.stack(output, dim=2)  
        return x

class Generator(nn.Module):
    def __init__(self, input_dim, ngf=64, output_dim=3, dropout=0):
        """
        DCGAN 生成器
        
        参数:
        - nz (int): 潜入向量z的维度
        - ngf (int): 生成器特征图的基础深度
        - nc (int): 输出图像的通道数
        """
        super(Generator, self).__init__()
        self.ngf = ngf  # 基础特征图深度
        output_dim = output_dim[0] # [C,H,W]
        self.kernal_size = 4  
        self.padding = int((self.kernal_size - 1) / 2) 

        self.l1 = nn.Sequential(
            self.fc(input_dim, ngf * 8 * 4 * 4, dropout), # (B, input_dim) -> (B, ngf * 8 * 4 * 4)
        )
        
        self.l2 = nn.Sequential(
            self.Iconv2d(ngf * 8, ngf * 4, 0),  # (B, ngf * 8, 4, 4) -> (B, ngf * 4, 8, 8)
            self.Iconv2d(ngf * 4, ngf * 2, 0),  # (B, ngf * 4, 8, 8) -> (B, ngf * 2, 16, 16)
            self.Iconv2d(ngf * 2, ngf, 0),      
        )

        self.l3 =  nn.ConvTranspose2d(in_channels=ngf, out_channels=output_dim, kernel_size=self.kernal_size, stride=2, padding=self.padding, bias=False)
        
        # self.l2 = nn.Sequential(
        #     self.upsample(ngf * 8, ngf * 4),  # (B, ngf * 8, 4, 4) -> (B, ngf * 4, 8, 8)
        #     self.upsample(ngf * 4, ngf * 2),
        #     self.upsample(ngf * 2, ngf),
        # )
        # self.l3 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(ngf, output_dim, kernel_size=5, stride=1, padding=2, bias=False)
        # )

    def Iconv2d(self, c_in, c_out, dropout=0):
        return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=self.kernal_size, stride=2, padding=self.padding, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout)
        )
    
    def upsample(self, c_in, c_out):
        return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(c_in, c_out, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True)
        )
    def fc(self, c_in, c_out, dropout=0):
        return nn.Sequential(
                nn.Linear(c_in, c_out),
                nn.BatchNorm1d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.ReLU(),
                nn.Dropout1d(dropout)
        )
    
    def forward(self, input):
        """
        前向传播
        
        参数:
        - input (Tensor): 输入的噪声向量，形状为 (B, nz)
        """

        x = self.l1(input)
        ngf = self.ngf
        x = x.view(-1, ngf * 8, 4, 4) # -1 表示自动计算 batch_size
        
        x = self.l2(x)
        output = self.l3(x)
        
        return output

class mymodel(nn.Module):
    def __init__(self, input_dim, ngf, output_feature, output_dim, dropout=0):
        super(mymodel, self).__init__()
        self.feat = PointNetEncoder(output_feature, global_feat=True, feature_transform=False, channel=2)
        self.generator = Generator(output_feature, ngf, output_dim, dropout)
    
    def forward(self, x):
        x = self.feat(x)
        x = self.generator(x)
        return x