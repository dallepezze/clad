import torch
import torch.nn as nn
#from einops import rearrange
#import numpy as np
#from tqdm import tqdm

#import torch.nn.functional as F
#from torchvision.transforms import InterpolationMode

def create_st(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    st = ST(parameters['model_size'], parameters['out_channels'], parameters['weights'], device)
    st = st.to(device)
    return st, device  


class ST(nn.Module):
    def __init__(self, model_size, out_channels, weights, device):
        super(ST, self).__init__()
        self.device = device
        self.model_size = model_size
        self.out_channels = out_channels
        if model_size == 'small':
            self.teacher = get_pdn_small(out_channels)
            self.student = get_pdn_small(out_channels)
        elif model_size == 'medium':
            self.teacher = get_pdn_medium(out_channels)
            self.student = get_pdn_medium(out_channels)
        else:
            raise Exception()
        
        self.training = True
        state_dict = torch.load(weights, map_location='cpu')#load the config.weights it is set by default to default='models/teacher_small.pth'
        #you can change it to 'models/teacher_medium.pth'
        self.teacher.load_state_dict(state_dict)#load the params in teacher model; training on ImageNet already done

        self.teacher_mean = torch.tensor(0)  
        self.teacher_std = torch.tensor(0)  

        self.teacher.eval()
        self.student.train()

    
    def forward(self,image_st, image_penalty):

        self.teacher.eval()

        if self.training == True:
            self.student.train()

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

            loss_total = loss_st 

            return loss_total
        
    def model_test(self, image_st, strategy):
        self.teacher.eval()
        self.student.eval()
    
        with torch.no_grad():
            map_st = predict(
                image=image_st, teacher=self.teacher, student=self.student,
                teacher_mean=self.teacher_mean,
                teacher_std=self.teacher_std, q_st_start=strategy.trainer.q_st_start, q_st_end=strategy.trainer.q_st_end)
            map_st = torch.nn.functional.pad(map_st, (4, 4, 4, 4))
            map_st = torch.nn.functional.interpolate(
                map_st, (image_st.shape[2], image_st.shape[2]), mode='bilinear')
            map_st = map_st.detach().cpu().numpy()
        
        return map_st
            


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
def predict(image, teacher, student,  teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None):
    
    teacher.eval()
    student.eval()

    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)

    map_st = torch.mean((teacher_output - student_output[:, :384])**2,
                        dim=1, keepdim=True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)

    return map_st