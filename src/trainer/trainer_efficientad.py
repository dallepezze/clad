import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
from tqdm import tqdm # this module is useful to plot progress bars
from typing import Callable, List, Tuple, Union, Iterable
import torch
import itertools
from torch import nn
import torch.nn.functional as F
from src.loss_functions import *
from src.utilities.utility_images import *
from src.utilities.utility_ad import standardize_scores, test_anomaly_maps, test_epoch_anomaly_maps

from src.utilities.utility_pix2pix import create_summary,create_summary_by_numpy


from adcl_paper.src.models.efficientad import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import sys

#Added
from src.utilities.utility_plot import*

class Trainer_efficientad():
    def __init__(self,strategy, eff):
        self.strategy = strategy
        self.device = strategy.device
        self.ad_model = eff
        self.batch_size = strategy.parameters['batch_size']
        self.lr = self.strategy.lr
        self.b1 = self.strategy.b1
        self.b2 = self.strategy.b2
        self.weight_decay = self.strategy.weight_decay
        self.optimizer = torch.optim.Adam(itertools.chain(self.ad_model.student.parameters(),
                                                 self.ad_model.autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
        
        self.pretrain_penalty = True
        if strategy.parameters['imagenet_train_path'] == 'None':
            self.pretrain_penalty = False


        if self.pretrain_penalty:
            # load pretraining data for penalty
            image_size = strategy.parameters['img_size']
            penalty_transform = transforms.Compose([
                transforms.Resize((2 * image_size, 2 * image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                    0.225])
            ])
            penalty_set = ImageFolderWithoutTarget(strategy.parameters['imagenet_train_path'],
                                                transform=penalty_transform)
            penalty_loader = DataLoader(penalty_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            self.penalty_loader_infinite = InfiniteDataloader(penalty_loader)
        else:
            self.penalty_loader_infinite = itertools.repeat(None)

        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(0.95 * strategy.parameters['num_epochs']), gamma=0.1)
    

    def train_epoch(self,dataloader):
        self.ad_model.training = True
        l_fastflow_loss = 0.0
        dataSize = len(dataloader.dataset)
        lista_indices = [] 
        self.ad_model.teacher.eval()
        self.ad_model.student.train()
        self.ad_model.autoencoder.train()

        batch_index = 0
        tqdm_obj = tqdm(range(dataSize))

        for _, batch, image_penalty in zip(tqdm_obj, dataloader, self.penalty_loader_infinite):
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            x = batch[0]
            batch_size = x.size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())

            image_st = x[:,0]
            image_ae = x[:,1]

            self.optimizer.zero_grad()
        
            image_st = image_st.to(self.ad_model.device)
            image_ae = image_ae.to(self.ad_model.device)
            image_penalty = image_penalty.to(self.ad_model.device)

            loss = self.ad_model(image_st, image_ae, image_penalty)

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            l_fastflow_loss += loss.item() * batch_size
            '''
            # only for debugging
            if  batch_index==(len(dataloader)-1):
                for i in range(1):
                    original_img = convert2img(x[i])
                    summary = create_summary_by_numpy([original_img])
            '''
            batch_index += 1

        '''
        torch.save(self.ad_model.teacher, os.path.join(self.strategy.train_output_dir,
                                            'teacher_tmp.pth'))
        torch.save(self.ad_model.student, os.path.join(self.strategy.train_output_dir,
                                            'student_tmp.pth'))
        torch.save(self.ad_model.autoencoder, os.path.join(self.strategy.train_output_dir,
                                                'autoencoder_tmp.pth'))
        '''
        if self.strategy.parameters['early_stopping'] == True:
            run_name1 = 'model_student'+str(self.strategy.current_epoch)
            run_name2 = 'model_autoencoder'+str(self.strategy.current_epoch)
            torch.save(self.ad_model.student.state_dict(), os.path.join(self.strategy.checkpoints, run_name1 + ".pckl"))
            torch.save(self.ad_model.autoencoder.state_dict(), os.path.join(self.strategy.checkpoints, run_name2 + ".pckl"))
            
            
        l_fastflow_loss /= dataSize
        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss":l_fastflow_loss}
        other_data_epoch = {"indices":lista_indices}
        
        return metrics_epoch,other_data_epoch   



    def test_epoch(self,dataloader):
        dataset = self.strategy.complete_test_dataset
        self.ad_model.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []
        batch_index = 0

        lista_indices = [] 
        self.ad_model.teacher.eval()
        self.ad_model.student.eval()
        self.ad_model.autoencoder.eval()

        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=dataloader, teacher=self.ad_model.teacher,
        student=self.ad_model.student, autoencoder=self.ad_model.autoencoder,
        teacher_mean=self.ad_model.teacher_mean, teacher_std=self.ad_model.teacher_std,
        desc='Intermediate map normalization')

        self.q_st_start, self.q_st_end, self.q_ae_start, self.q_ae_end = q_st_start, q_st_end, q_ae_start, q_ae_end

        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            ###
            image_st = batch[0][:,0]
            #print(image_st.shape)
            image_st = image_st.to(self.ad_model.device)
            
            with torch.no_grad():
                map_combined, map_st, map_ae = predict(
                    image=image_st, teacher=self.ad_model.teacher, student=self.ad_model.student,
                    autoencoder=self.ad_model.autoencoder, teacher_mean=self.ad_model.teacher_mean,
                    teacher_std=self.ad_model.teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
                    q_ae_start=q_ae_start, q_ae_end=q_ae_end)
                map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
                map_combined = torch.nn.functional.interpolate(
                    map_combined, (image_st.shape[2], image_st.shape[2]), mode='bilinear')
                map_combined = map_combined[:, 0].detach().cpu().numpy()
                
            ###
            l_anomaly_maps.extend(map_combined)

            lista_labels.extend(class_ids)
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path,anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            #test_imgs.extend(data.cpu().numpy())
            gt_list.extend(anomaly_info.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())

            batch_index +=1
            
        l_anomaly_maps = standardize_scores(l_anomaly_maps)
        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"
        
        metrics_epoch = diz_metriche
        other_data_epoch = {}

        
        return metrics_epoch, other_data_epoch

    def evaluate_data(self, dataloader,test_loss_function=None):  
        dataset = self.strategy.complete_test_dataset
        self.ad_model.training = False
        lista_indices = []
        losses, l_anomaly_maps, lista_labels = [], [], []
        test_imgs, gt_list, gt_mask_list = [], [], []
        batch_index = 0

        lista_indices = [] 
        self.ad_model.teacher.eval()
        self.ad_model.student.eval()
        self.ad_model.autoencoder.eval()


        #self.trainer.vae.teacher_mean, self.trainer.vae.teacher_std = teacher_normalization(self.trainer.vae.teacher, dataloader)

        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=dataloader, teacher=self.ad_model.teacher,
        student=self.ad_model.student, autoencoder=self.ad_model.autoencoder,
        teacher_mean=self.ad_model.teacher_mean, teacher_std=self.ad_model.teacher_std,
        desc='Intermediate map normalization')


        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            #test_imgs.extend(data.detach().cpu().numpy())
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            ###
            image_st = batch[0][:,0]
            test_imgs.extend(image_st.detach().numpy())
            image_st = image_st.to(self.ad_model.device)

            with torch.no_grad():
                map_combined, map_st, map_ae = predict(
                    image=image_st, teacher=self.ad_model.teacher, student=self.ad_model.student,
                    autoencoder=self.ad_model.autoencoder, teacher_mean=self.ad_model.teacher_mean,
                    teacher_std=self.ad_model.teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
                    q_ae_start=q_ae_start, q_ae_end=q_ae_end)
                map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
                map_combined = torch.nn.functional.interpolate(
                    map_combined, (image_st.shape[2], image_st.shape[2]), mode='bilinear')
                map_combined = map_combined[:,0].detach().cpu().numpy()
                        
            ###
            l_anomaly_maps.extend(map_combined)

            lista_labels.extend(class_ids.detach().cpu().numpy())
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path,anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            #test_imgs.extend(data.cpu().numpy())
            gt_list.extend(anomaly_info.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())

            batch_index +=1
            
        l_anomaly_maps = standardize_scores(l_anomaly_maps)#np (total num of samples, 256, 256)
        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        #added
        if self.strategy.index_training == 9:
            plot_predict(self, lista_labels, l_anomaly_maps, gt_mask_list, lista_indices, threshold, test_imgs)
        

        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"
        
        metrics_epoch = diz_metriche
        other_data_epoch = {}

        
        return metrics_epoch, other_data_epoch
    



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



@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    
    teacher.eval()
    student.eval()
    autoencoder.eval()

    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for batch in tqdm(validation_loader, desc=desc):
        if torch.cuda.is_available():
            image = batch[0][:,0].to("cuda:0")
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end




@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    teacher.eval()

    mean_outputs = []
    for batch in tqdm(train_loader, desc='Computing mean of features'):
        if torch.cuda.is_available():
            train_image = batch[0][:,0].to(teacher.device)
            print("Image device: " + str(train_image.device))
            sys.exit()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if torch.cuda.is_available():
            train_image = batch[0][:,0].to("cuda:0")
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
