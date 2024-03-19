import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from src.models.cfa_add.metric import *
from src.models.cfa_add.coordconv import CoordConv2d
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


#from src.models.cfa_add.cnn.resnet import resnet18 as res18
#from src.models.cfa_add.cnn.efficientnet import EfficientNet as effnet
#from src.models.cfa_add.cnn.vgg import vgg19_bn as vgg19
'''
def create_cae_model(strategy, img_shape, parameters):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cae = CAE(img_shape,parameters['latent_dim'])
    cae = cae.to(device)
    return cae, device  


########## architecture 1 with 6 conv layers with size 256 x 256 #################

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True,dropoutvalue=0):
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        )
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropoutvalue!=0:
            layers.append(nn.Dropout(dropoutvalue))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv_block(x)
        return x

class UpBlock(nn.Module): 
    def __init__(self,in_channels, out_channels,kernel_size, stride, padding, batchnorm=True,dropoutvalue=0):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if batchnorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        if dropoutvalue!=0:
            layers.append(nn.Dropout(dropoutvalue))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self,x):
       x = self.conv_block(x)
       return x 

class Encoder(nn.Module):
    def __init__(self, latent_dim,n_features = 16):
        super().__init__()
        self.down1 = DownBlock(3,n_features,4,2,1,batchnorm=False,dropoutvalue=False) # 128x128
        self.down2 = DownBlock(n_features,n_features*2,4,2,1,batchnorm=True,dropoutvalue=False) # 64x64
        self.down3 = DownBlock(n_features*2,n_features*4,4,2,1,batchnorm=True,dropoutvalue=False) #32x32
        self.down4 = DownBlock(n_features*4,n_features*8,4,2,1,batchnorm=True,dropoutvalue=False) #16x16
        self.down5 = DownBlock(n_features*8,n_features*16,4,2,1,batchnorm=True,dropoutvalue=False) #8x8
        self.down6 = DownBlock(n_features*16,latent_dim,4,2,1,batchnorm=False,dropoutvalue=False) #4x4x512 

    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        z = self.down6(x)
        return z

class Decoder(nn.Module):
    def __init__(self,img_shape, latent_dim,n_features=16):
        super().__init__()
        self.img_shape = img_shape
        self.up1 = UpBlock(latent_dim,n_features*16,4,2,1,batchnorm=False,dropoutvalue=False)
        self.up2 = UpBlock(n_features*16,n_features*8,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up3 = UpBlock(n_features*8,n_features*4,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up4 = UpBlock(n_features*4,n_features*2,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up5 = UpBlock(n_features*2,n_features,4,2,1,batchnorm=True,dropoutvalue=False)
        self.up6 = nn.Sequential(nn.ConvTranspose2d(n_features,3,4,2,padding=1),
                   nn.Tanh())

    def forward(self,z):
        img = self.up1(z)
        img = self.up2(img)
        img = self.up3(img)
        img = self.up4(img)
        img = self.up5(img)
        img = self.up6(img)
        return img.view(img.shape[0], *self.img_shape)


class CAE(nn.Module):
    def __init__(self,img_shape, latent_dim):
        super(CAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(img_shape, latent_dim)

    def forward(self,x):
        z  = self.encoder(x)
        x_hat =  self.decoder(z)
        return x_hat,z,None,None'''

def create_cfa(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    cfa = DSVDD(parameters['gamma_c'], parameters['gamma_d'], device)
    cfa = cfa.to(device)
    return cfa, device  


class DSVDD(nn.Module):
    def __init__(self, gamma_c, gamma_d, device):
        super(DSVDD, self).__init__()
        self.device = device
        self.C = torch.tensor(0)
        self.D = torch.tensor(0)  
        self.nu = 1e-3
        self.scale = None
        self.cnn = 'wrn50_2'
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3
        self.total_num_batches = 0

        self.r   = nn.Parameter(1e-5*torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d, self.cnn)
        #self._init_centroid(model, data_loader)


    def forward(self, p, model):
        model.eval()
        x = model(p)
        phi_p = self.Descriptor(x)       
        phi_p = rearrange(phi_p, 'b c h w -> b (h w) c') #dim (batch, resolution, channel depth)
        
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)  #dim (batch, resolution, 1)  
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True) #dim (1, (batch,resolution))
        f_c      = 2 * torch.matmul(phi_p, (self.C)) 
        dist     = features + centers - f_c
        dist     = torch.sqrt(dist)

        n_neighbors = self.K
        dist     = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        score = rearrange(dist, 'b (h w) c -> b c h w', h=self.scale)
        
        loss = 0
        if self.training:
            loss = self._soft_boundary(phi_p)

        return loss, score

    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers  = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c      = 2 * torch.matmul(phi_p, (self.C))
        dist     = features + centers - f_c
        n_neighbors = self.K + self.J
        dist     = dist.topk(n_neighbors, largest=False).values

        score = (dist[:, : , :self.K] - self.r**2) 
        L_att = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))
        
        score = (self.r**2 - dist[:, : , self.J:]) 
        L_rep  = (1/self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))
        
        loss = L_att + L_rep

        return loss 

    def _update_centroid(self, data_loader, model):
        k = 0
        model.eval()
        for i, (x, _, _,_,_) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
                        
            phi_p = self.Descriptor(p) #dim: (batch size =4, channel depth, H, W)
            j = i + self.total_num_batches
            k = i
            self.D = (((self.D * j)) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (j+1) #keepdim=True-> dim_out: (batch size=1, channel depth, H, W), #incremental average
        k = k + 1
        
        p1 = self.D.device
        print("Device:", p1)
        if self.total_num_batches>0:
            self.C = nn.Parameter(torch.empty(0).cuda())
            self.C.data = rearrange(self.D, 'b c h w -> (b h w) c').transpose(-1, -2).detach().clone()
 
        
        if self.total_num_batches==0:
            self.C = self.D.detach().clone()
            self.C = rearrange(self.C, 'b c h w -> (b h w) c').detach()
            
            if self.gamma_c > 1:
                self.C = self.C.cpu().detach().numpy()
                self.C = KMeans(n_clusters=(self.scale**2)//self.gamma_c, max_iter=3000).fit(self.C).cluster_centers_
                self.C = torch.Tensor(self.C).to(self.device)
    
            self.C = self.C.transpose(-1, -2).detach()#dim (channel depth, sth)
            self.C = nn.Parameter(self.C, requires_grad=False)
        p2 = self.C.device
        print("Device:", p2)
        self.total_num_batches += k
        
        #print dimension of the memorized patch features
        print("dimension of the memorized patch features:", self.C.shape)

class Descriptor(nn.Module):
    def __init__(self, gamma_d, cnn):#cnn is a string
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == 'wrn50_2':
            dim = 1792 
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        elif cnn == 'res18':
            dim = 448
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        elif cnn == 'effnet-b5':
            dim = 568
            self.layer = CoordConv2d(dim, 2*dim//gamma_d, 1)
        elif cnn == 'vgg19':
            dim = 1280 
            self.layer = CoordConv2d(dim, dim//gamma_d, 1)
        

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == 'effnet-b5' else F.avg_pool2d(o, 3, 1, 1)
            sample = o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode='bilinear')), dim=1)
        # sample dim after for loop: #dim: (batch size=4, channels=1792, H, W)
        phi_p = self.layer(sample) #dim: (batch size=4, channels=1792, H, W)
        return phi_p #dim: (batch size=4, channels=1792, H, W)