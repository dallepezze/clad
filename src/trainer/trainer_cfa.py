import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
from tqdm import tqdm # this module is useful to plot progress bars
from typing import Callable, List, Tuple, Union, Iterable
import torch
from torch import nn
import torch.nn.functional as F
from src.loss_functions import *
from src.utilities.utility_images import *
from src.utilities.utility_ad import standardize_scores, test_anomaly_maps, test_epoch_anomaly_maps

from src.utilities.utility_pix2pix import create_summary,create_summary_by_numpy


from src.models.cfa_add.metric import *
from src.models.cfa_add.visualizer import * 

from src.models.cfa import *

#Added
from src.utilities.utility_plot import*

class Trainer_cfa():
    def __init__(self,strategy, cfa):
        self.strategy = strategy
        #self.ad_model = cfa.to(strategy.device)
        self.ad_model = cfa
        self.lr = self.strategy.lr
        self.b1 = self.strategy.b1
        self.b2 = self.strategy.b2
        self.weight_decay = self.strategy.weight_decay
        self.optimizer = torch.optim.AdamW(self.ad_model.parameters(), lr=1e-3, weight_decay= 5e-4, amsgrad = True)
        #self.optimizer = torch.optim.Adam(self.ad_model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay= 5e-4)
        #weight_decay= self.weight_decay
    
    def train_epoch(self,dataloader):
        self.ad_model.training = True
        l_fastflow_loss = 0.0
        dataSize = len(dataloader.dataset)
        lista_indices = [] 
        self.strategy.model.eval()
        self.ad_model.train()

        batch_index = 0
        for batch in tqdm(dataloader):
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            x = batch[0]
            batch_size = x.size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())

            self.optimizer.zero_grad()

            x = x.to(self.ad_model.device)
            loss, score = self.ad_model(x, self.strategy.model)

            loss.backward()
            self.optimizer.step()
            l_fastflow_loss += loss.item() * batch_size

            # only for debugging
            if  batch_index==(len(dataloader)-1):
                for i in range(1):
                    original_img = convert2img(x[i])
                    summary = create_summary_by_numpy([original_img])

            batch_index += 1
            
        l_fastflow_loss /= dataSize
        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss":l_fastflow_loss}
        other_data_epoch = {"indices":lista_indices}
        
        return metrics_epoch,other_data_epoch   

    def test_epoch(self,dataloader):
        dataset = self.strategy.complete_test_dataset
        self.ad_model.training = False
        lista_indices = []
        losses, heatmaps, lista_labels = [], None, []
        test_imgs, gt_list, gt_mask_list = [], [], []
        batch_index = 0
        max_score_value = 0
        min_score_value = -10000000

        self.strategy.model.eval()
        self.ad_model.eval()
        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            data = data.to(self.ad_model.device)
            if self.strategy.parameters['architecture']=='cfa':
                loss, anomaly_maps = self.ad_model(data, self.strategy.model)
            else:
                with torch.no_grad():

                    loss, anomaly_maps = self.ad_model(data, self.strategy.model)#added

            heatmap = anomaly_maps.cpu().detach()
            heatmap = torch.mean(heatmap, dim=1) 
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            
            #lista_labels.extend(class_ids)
            lista_labels.extend(class_ids.detach().cpu().numpy())
            
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)

            mask = torch.stack(masks)
            test_imgs.extend(data.detach().cpu().numpy())
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().cpu().numpy())

            batch_index +=1

        heatmaps = upsample(heatmaps, size=data.size(2), mode='bilinear')
        heatmaps = gaussian_smooth(heatmaps, sigma=4)
    
        #gt_mask = np.asarray(gt_mask_list)
        scores = rescale(heatmaps)

        scores = scores 
        #threshold = get_threshold(gt_mask, scores)

        diz_metriche = test_epoch_anomaly_maps(scores,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]
        
        mode = self.strategy.trainer.mode if hasattr(self.strategy.trainer, 'mode') else "reconstruct"
        
        metrics_epoch = diz_metriche
        other_data_epoch = {}

        
        return metrics_epoch, other_data_epoch

    def evaluate_data(self, dataloader,test_loss_function=None):  
        dataset = self.strategy.complete_test_dataset
        test_task_index = self.strategy.current_test_task_index
        index_training = self.strategy.index_training
        self.ad_model.training = False
        lista_indices = []
        losses, heatmaps, lista_labels = [], None, []
        test_imgs, gt_list, gt_mask_list = [], [], []

        self.strategy.model.eval()
        self.ad_model.eval()
        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            #class_ids = batch[1]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            data = data.to(self.ad_model.device)

            if self.strategy.parameters['architecture']=='cfa':
                loss, anomaly_maps = self.ad_model(data, self.strategy.model)
            else:
                with torch.no_grad():

                    loss, anomaly_maps = self.ad_model(data, self.strategy.model)#added

            heatmap = anomaly_maps.cpu().detach()
            heatmap = torch.mean(heatmap, dim=1) 
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            
            lista_labels.extend(class_ids.detach().cpu().numpy())
            
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            test_imgs.extend(data.detach().cpu().numpy())
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().cpu().numpy())


        heatmaps = upsample(heatmaps, size=data.size(2), mode='bilinear')
        heatmaps = gaussian_smooth(heatmaps, sigma=4)
    
        #gt_mask = np.asarray(gt_mask_list)
        scores = rescale(heatmaps)

        scores = scores 
        #threshold = get_threshold(gt_mask, scores)

        diz_metriche = test_epoch_anomaly_maps(scores,gt_mask_list, gt_list, self.strategy.index_training, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]

        #added
        if self.strategy.index_training == 9:
            plot_predict(self, lista_labels, scores, gt_mask_list, lista_indices, threshold, test_imgs)
                
        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch