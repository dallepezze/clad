import torch
import torch.nn as nn
#from einops import rearrange
#import numpy as np
#from tqdm import tqdm

#import torch.nn.functional as F
#from torchvision.transforms import InterpolationMode

def create_efficientad(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    eff = EfficientAD(parameters['model_size'], parameters['out_channels'], parameters['weights'], device)
    eff = eff.to(device)
    return eff, device  


class EfficientAD(nn.Module):
    def __init__(self, model_size, out_channels, weights, device):
        super(EfficientAD, self).__init__()
        self.device = device
        self.model_size = model_size
        self.out_channels = out_channels
        if model_size == 'small':
            self.teacher = get_pdn_small(out_channels)
            self.student = get_pdn_small(2 * out_channels)
        elif model_size == 'medium':
            self.teacher = get_pdn_medium(out_channels)
            self.student = get_pdn_medium(2 * out_channels)
        else:
            raise Exception()
        
        self.training = True
        state_dict = torch.load(weights, map_location='cpu')#load the config.weights it is set by default to default='models/teacher_small.pth'
        #you can change it to 'models/teacher_medium.pth'
        self.teacher.load_state_dict(state_dict)#load the params in teacher model; training on ImageNet already done
        self.autoencoder = get_autoencoder(out_channels)#define autoencoder

        self.teacher_mean = torch.tensor(0)  
        self.teacher_std = torch.tensor(0)  

        self.teacher.eval()
        self.student.train()
        self.autoencoder.train()

    
    def forward(self,image_st,image_ae,image_penalty):

        self.teacher.eval()

        if self.training == True:
            self.student.train()
            self.autoencoder.train()

            with torch.no_grad():
                teacher_output_st = self.teacher(image_st)
                teacher_output_st = (teacher_output_st - self.teacher_mean) / self.teacher_std
            student_output_st = self.student(image_st)[:, :self.out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
    
            if image_penalty is not None:
                student_output_penalty = self.student(image_penalty)[:, :self.out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard
    
            ae_output = self.autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = self.teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std
            student_output_ae = self.student(image_ae)[:, self.out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            return loss_total
        
    def model_test(self, image_st, strategy):
        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()
    
        with torch.no_grad():
            map_combined, map_st, map_ae = predict(
                image=image_st, teacher=self.teacher, student=self.student,
                autoencoder=self.autoencoder, teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std, q_st_start=strategy.trainer.q_st_start, q_st_end=strategy.trainer.q_st_end,
                q_ae_start=strategy.trainer.q_ae_start, q_ae_end=strategy.trainer.q_ae_end)
            map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            map_combined = torch.nn.functional.interpolate(
                map_combined, (image_st.shape[2], image_st.shape[2]), mode='bilinear')
            map_combined = map_combined.detach().cpu().numpy()
        
        return map_combined
            


def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )


@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher.eval()
    student.eval()
    autoencoder.eval()
    
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :384])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, 384:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae