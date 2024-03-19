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

from src.models.padim import *

#Added
from src.utilities.utility_plot import*

class Trainer_padim():
    def __init__(self,strategy, padim):
        self.strategy = strategy
        self.ad_model = padim
        self.d_reduced = strategy.parameters["d_reduced"]
        self.device = self.ad_model.device
        self.MEAN = []
        self.COV = []
        self.r_indices = None

    def train_epoch(self,dataloader):
        dataSize = len(dataloader.dataset)
        lista_indices = [] 

        patch_lib = []
        batch_index = 0
        for batch in tqdm(dataloader):
            batch = self.strategy.memory.create_batch_data(batch, batch[0].shape[0])
            x = batch[0]
            batch_size = x.size(0)
            indices = batch[2]
            lista_indices.extend(indices.detach().cpu().numpy())
            
            
            x = x.to(self.ad_model.device)
            self.strategy.model.eval()
            feature_maps = self.ad_model(x, self.strategy.model)

            '''
            # only for debugging
            if  batch_index==(len(dataloader)-1):
                for i in range(1):
                    original_img = convert2img(x[i])
                    summary = create_summary_by_numpy([original_img])
            '''
            batch_index += 1

            if self.ad_model.resize is None:
                largest_fmap_size = feature_maps[0].shape[-2:]
                self.ad_model.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
            resized_maps = [self.ad_model.resize(fmap) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            #print(f'Batch patch size: {patch.shape}')
            patch_lib.append(patch)

        #print(f'Num of batches: {batch_index}')
        patch_lib = torch.cat(patch_lib, 0)
        print(f"Size of patch_lib: {patch_lib.shape}")

		# random projection
        if patch_lib.shape[1] > self.d_reduced:
            print(f"   PaDiM: (randomly) reducing {patch_lib.shape[1]} dimensions to {self.d_reduced}.")
            if self.r_indices == None:
                self.r_indices = torch.randperm(patch_lib.shape[1])[:self.d_reduced]
            patch_lib_reduced = patch_lib[:,self.r_indices,...]
        else:
            print("   PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
            patch_lib_reduced = patch_lib


        # mean calc
        means = torch.mean(patch_lib, dim=0, keepdim=True)#(1,num_channels,...)
        means_reduced = means[:,self.r_indices,...]#(1,rd_550,56,56)
        x_ = patch_lib_reduced - means_reduced#(1,rd_550,56,56)

        # cov calc
        E = torch.einsum('abkl,bckl->ackl',x_.permute([1,0,2,3]),x_) * 1/(patch_lib.shape[0]-1)
        E += self.ad_model.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
        E_inv = torch.linalg.inv(E.permute([2,3,0,1])).permute([2,3,0,1])#(rd_550,rd_550,56,56)


        
        if self.strategy.parameters["cl"]:
            
            if self.strategy.index_training == 0 and self.strategy.parameters["sample_strategy"] in ["cl", "naive", "single_model", "multi_task", "replay"]:
                #print(f"Means_reduced shape each: {means_reduced.shape}")
                #self.MEAN.append(means_reduced.detach().cpu())
                self.MEAN.append(means_reduced.detach())
                print(f"Mean shape: {means_reduced.shape}")
                #print(f"Means_reduced shape each: {E_inv.shape}")
                #self.COV.append(E_inv.detach().cpu())
                self.COV.append(E_inv.detach())
                print(f"Cov shape: {E_inv.shape}")
            
            else:
                #print(f"Means_reduced shape each: {means_reduced.shape}")
                self.MEAN[0] = (self.MEAN[0]*self.strategy.index_training+means_reduced)/(self.strategy.index_training+1)
                #print(f"Means_reduced shape each: {E_inv.shape}")
                self.COV[0] = (self.COV[0]*self.strategy.index_training+E_inv)/(self.strategy.index_training+1)
                
        else:
            #self.MEAN.append(means_reduced.detach().to(self.ad_model.device))
            #self.COV.append(E_inv.detach().to(self.ad_model.device))
            self.MEAN.append(means_reduced.detach())
            self.COV.append(E_inv.detach())
            
        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss":0}
        other_data_epoch = {"indices":lista_indices}
        
        del E, E_inv, means, means_reduced, x_, patch, patch_lib, patch_lib_reduced 

        return metrics_epoch,other_data_epoch   

    def test_epoch(self,dataloader):
        return None, None

    def evaluate_data(self, dataloader,test_loss_function=None):  
        dataset = self.strategy.complete_test_dataset
        test_task_index = self.strategy.current_test_task_index
        index_training = self.strategy.index_training
        lista_indices = []
        losses, heatmaps, lista_labels = [], None, []
        test_imgs, gt_list, gt_mask_list = [], [], []
        image_preds = []

        dataSize = len(dataloader.dataset)
        precision = 0

        for batch in tqdm(dataloader):
            masks = []
            data,indices,anomaly_info= batch[0],batch[2],batch[3]
            #class_ids = batch[1]
            class_ids = batch[1]
            lista_indices.extend(batch[2].detach().cpu().numpy()) 
            test_imgs.extend(data.detach().numpy())
            data = data.to(self.ad_model.device)

            self.strategy.model.eval()
            feature_maps = self.ad_model(data,self.strategy.model)
            resized_maps = [self.ad_model.resize(fmap) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            
            #test patch (1,num_channels,...)
            #patch = patch.detach().to(self.ad_model.device)
            
            #set minimum (for loop)
            minimum = 10000000000
            index_task = -1     
            

            for i in range(len(self.MEAN)):
                x_ = patch[:,self.r_indices,...] - self.MEAN[i]
                left = torch.einsum('abkl,bckl->ackl', x_, self.COV[i])
                s_map1 = torch.tensor(0)
                s_map1 = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))#(1,...)    
                suma = torch.tensor(0)
                suma = torch.sum(s_map1)
                if suma<minimum:
                    minimum = suma 
                    index_task = i
                    #s_map = s_map1.detach().cpu()
                    s_map = s_map1.detach()
                    
                del x_, left, s_map1

            del patch, resized_maps 

            #print(f"Chosen class:{index_task}")
            if index_task == test_task_index:
                precision += 1

            #image_preds.append(torch.max(s_map).detach().cpu().numpy())
            image_preds.append(torch.max(s_map).detach().numpy())
            
            #heatmap = s_map.detach().cpu()
            heatmap = s_map.detach()
            
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            #print(f"heatmaps after torch.cat: {heatmaps.shape}")
            lista_labels.extend(class_ids.detach().cpu().numpy())
            
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            #test_imgs.extend(data.detach().cpu().numpy())
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            #gt_mask_list.extend(mask.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().numpy())

            del s_map, mask, masks
        

        self.strategy.run.log({f"Task_average_precision/T{index_training}": precision/dataSize})
        #self.strategy.run.log({f"Task_average_precision": precision/dataSize})

        heatmaps = upsample(heatmaps, size=self.ad_model.image_size, mode='bilinear')
        heatmaps = gaussian_smooth(heatmaps, sigma=4)
    
        #gt_mask = np.asarray(gt_mask_list)
        scores = rescale(heatmaps)
        image_preds = np.stack(image_preds)
        image_preds = rescale(image_preds) 
        #scores = np.asarray(heatmaps)
        #print(f"scores shape final:{scores.shape}")

        diz_metriche = test_epoch_anomaly_maps(scores,gt_mask_list, gt_list, image_preds, self.strategy.run, self.strategy.labels_map[self.strategy.index_training],self.strategy.index_training,self.strategy.path_logs)
        diz_metriche["loss"] = 1-diz_metriche["per_pixel_rocauc"]
        threshold = diz_metriche["threshold"]

        #added
        if self.strategy.index_training == 9:
            plot_predict(self, lista_labels, scores, gt_mask_list, lista_indices, threshold, test_imgs)
        
        del scores, heatmaps

        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch
    