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

from src.models.cfa_add.metric import *
from src.models.cfa_add.visualizer import * 

from src.models.draem import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

#Added
from src.utilities.utility_plot import*

class Trainer_draem():
    def __init__(self,strategy, draem):
        self.strategy = strategy
        self.device = strategy.device
        self.ad_model = draem
        self.batch_size = strategy.parameters['batch_size']
        self.lr = self.strategy.lr
        self.num_epochs = self.strategy.parameters['num_epochs']
        #self.b1 = self.strategy.b1
        #self.b2 = self.strategy.b2
        #self.weight_decay = self.strategy.weight_decay
        self.optimizer = torch.optim.Adam([
                                      {"params": self.ad_model.model.parameters(), "lr": self.lr},
                                      {"params": self.ad_model.model_seg.parameters(), "lr": self.lr}])
        
        #added
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[self.num_epochs*0.8,self.num_epochs*0.9],gamma=0.2, last_epoch=-1)


    def train_epoch(self,dataloader):
        self.ad_model.training = True
        l_fastflow_loss = 0.0
        dataSize = len(dataloader.dataset)
        lista_indices = [] 
        self.ad_model.model.train()
        self.ad_model.model_seg.train()

        batch_index = 0

        for batch in tqdm(dataloader):
            batch = self.strategy.memory.create_batch_data(batch, batch[0]['image'].shape[0])
            #shapee = batch[0]['image'].shape
            #print(f'Batch train shape:{shapee}')
            #x = batch[0]
            batch_size = batch[0]['image'].size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())

            self.optimizer.zero_grad()

            gray_batch = batch[0]["image"].to(self.ad_model.device)
            aug_gray_batch = batch[0]["augmented_image"].to(self.ad_model.device)
            anomaly_mask = batch[0]["anomaly_mask"].to(self.ad_model.device)

            loss = self.ad_model(gray_batch, aug_gray_batch, anomaly_mask)

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
        #added
        self.scheduler.step()        
        if self.strategy.parameters['early_stopping'] == True:
            run_name1 = 'model_'+str(self.strategy.current_epoch)
            run_name2 = 'model_seg_'+str(self.strategy.current_epoch)
            torch.save(self.ad_model.model.state_dict(), os.path.join(self.strategy.checkpoints, run_name1 + ".pckl"))
            torch.save(self.ad_model.model_seg.state_dict(), os.path.join(self.strategy.checkpoints, run_name2 + ".pckl"))

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
        self.ad_model.model.eval()
        self.ad_model.model_seg.eval()
        image_preds = []
        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            ###
            data = data.to(self.ad_model.device)

            heatmap, anomaly_score = self.ad_model.model_test(data)#tensor(1,256,256), tensor(1,)
            image_preds.append(anomaly_score)
    
            ###
            l_anomaly_maps.extend(heatmap)

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
        l_anomaly_maps = np.asarray(l_anomaly_maps)
        image_preds = np.stack(image_preds)
        image_preds = rescale(image_preds) 


        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, image_preds, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
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
        self.ad_model.model.eval()
        self.ad_model.model_seg.eval()
        image_preds = []

        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            test_imgs.extend(data.detach().numpy())
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            ###
            data = data.to(self.ad_model.device)

            heatmap, anomaly_score = self.ad_model.model_test(data)#numpy(1,256,256), numpy(1,)
            #added
            
            image_preds.append(anomaly_score)
    
            ###
            l_anomaly_maps.extend(heatmap)

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
            
        l_anomaly_maps = np.asarray(l_anomaly_maps)
        #commented
        #l_anomaly_maps = gaussian_smooth(l_anomaly_maps, sigma=2)
        #l_anomaly_maps = standardize_scores(l_anomaly_maps)

        #l_anomaly_maps = np.asarray(l_anomaly_maps)
        image_preds = np.stack(image_preds)
        #commented
        #image_preds = rescale(image_preds) 

        diz_metriche = test_epoch_anomaly_maps(l_anomaly_maps,gt_mask_list, gt_list, image_preds, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        #added
        if self.strategy.index_training == 9:
            plot_predict_draem(self, lista_labels, l_anomaly_maps, gt_mask_list, lista_indices, threshold, test_imgs)
        

        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"
        
        metrics_epoch = diz_metriche
        other_data_epoch = {}

        
        return metrics_epoch, other_data_epoch