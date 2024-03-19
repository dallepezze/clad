from xmlrpc.client import FastMarshaller
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

from adcl_paper.src.models.patchcore import *

#Added
from src.utilities.utility_plot import*

class Trainer_patchcore():
    def __init__(self,strategy, patch):
        self.strategy = strategy
        self.ad_model = patch
    
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
            resized_maps = [self.ad_model.resize(self.ad_model.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            patch = patch.reshape(patch.shape[1], -1).T
            #print(f'Batch patch size: {patch.shape}')
            patch_lib.append(patch)

        #print(f'Num of batches: {batch_index}')
        patch_lib = torch.cat(patch_lib, 0)
        #print(f"Size of patch_lib before entering get_coreset_idx_randomp: {patch_lib.shape}")
        
        mem_size = self.strategy.parameters['mem_size_cl']
        
        if self.strategy.parameters['cl'] == False:
            if self.ad_model.f_coreset < 1:
                coreset_idx = get_coreset_idx_randomp(patch_lib, n=int(self.ad_model.f_coreset * patch_lib.shape[0]),eps=self.ad_model.coreset_eps)
                patch_lib = patch_lib[coreset_idx]
                self.ad_model.list_mem.append(patch_lib.detach().to(self.ad_model.device))
        else:
            for i in range(len(self.ad_model.list_mem)):
                coreset_idx = get_coreset_idx_randomp(self.ad_model.list_mem[i],  n=int(mem_size/(self.strategy.index_training+1)),eps=self.ad_model.coreset_eps)
                self.ad_model.list_mem[i] = self.ad_model.list_mem[i][coreset_idx]

            coreset_idx = get_coreset_idx_randomp(patch_lib, n=int(mem_size/(self.strategy.index_training+1)),eps=self.ad_model.coreset_eps)
            patch_lib = patch_lib[coreset_idx]
            self.ad_model.list_mem.append(patch_lib.detach().to(self.ad_model.device))


        lista_indices = np.asarray(lista_indices)

        metrics_epoch = {"loss":0}
        other_data_epoch = {"indices":lista_indices}
        
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
            lista_indices.extend(batch[2].detach().numpy()) 
            test_imgs.extend(data.detach().numpy())
            data = data.to(self.ad_model.device)

            self.strategy.model.eval()
            feature_maps = self.ad_model(data,self.strategy.model)
            resized_maps = [self.ad_model.resize(self.ad_model.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)
            patch = patch.reshape(patch.shape[1], -1).T
            
            #added
            patch = patch.to(self.ad_model.device)
            
            minimum = 10000000000
            index_task = -1

            #print(f"memory len:len(self.ad_model.list_mem)")
            for i in range(len(self.ad_model.list_mem)):
                dist = torch.tensor(0)
                dist = torch.cdist(patch, self.ad_model.list_mem[i])
                min_val1, min_idx1 = torch.tensor(0),torch.tensor(0)
                min_val1, min_idx1 = torch.min(dist, dim=1)
                s_idx1 = torch.argmax(min_val1)
                s_star1 = torch.max(min_val1)
                if torch.mean(min_val1)<minimum:
                     minimum = torch.mean(min_val1).clone()
                     s_star = s_star1.clone().to(self.ad_model.device)
                     index_task = i
                     min_val = min_val1.clone().to(self.ad_model.device)
                     min_idx = min_idx1.clone().to(self.ad_model.device)
                     s_idx = s_idx1.clone().to(self.ad_model.device)

            #print(f"Chosen class:{index_task}")
            if index_task == test_task_index:
                precision += 1

            # reweighting
            m_test = patch[s_idx].unsqueeze(0) # anomalous patch
            m_star = self.ad_model.list_mem[index_task][min_idx[s_idx]].unsqueeze(0) # closest neighbour
            w_dist = torch.cdist(m_star, self.ad_model.list_mem[index_task]) # find knn to m_star pt.1
            _, nn_idx = torch.topk(w_dist, k=self.ad_model.n_reweight, largest=False) # pt.2
            # equation 7 from the paper
            m_star_knn = torch.linalg.norm(m_test-self.ad_model.list_mem[index_task][nn_idx[0,1:]], dim=1)
            # Softmax normalization trick as in transformers.
            # As the patch vectors grow larger, their norm might differ a lot.
            # exp(norm) can give infinities.
            D = torch.sqrt(torch.tensor(patch.shape[1]))
            w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
            s = w*s_star
            image_preds.append(s.detach().cpu().numpy())
            
            # segmentation map
            s_map = min_val.detach().cpu().view(1,*feature_maps[0].shape[-2:])#(1,28,28)
            #print(f"s_map after min_val:{s_map.shape}")
            '''s_map = torch.nn.functional.interpolate(
                    s_map, size=(self.ad_model.image_size,self.ad_model.image_size), mode='bilinear'
            )'''
            #print(f"s_map after interpolation:{s_map.shape}")
            '''s_map = self.ad_model.blur(s_map)'''
            #print(f"s_map after blur:{s_map.shape}")
            heatmap = s_map
            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            #print(f"heatmaps after torch.cat: {heatmaps.shape}")
            lista_labels.extend(class_ids.detach().cpu().numpy())
            
            for i,idx in enumerate(indices):
                mask_path = dataset.mask[idx]
                mask = dataset.get_mask(mask_path, anomaly_info[i])
                masks.append(mask)
            mask = torch.stack(masks)
            gt_list.extend(anomaly_info.detach().cpu().numpy())
            gt_mask_list.extend(mask.detach().numpy())


        self.strategy.run.log({f"Task_average_precision/T{index_training}": precision/dataSize})

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
                
        metrics_epoch = diz_metriche
        other_data_epoch = {}
        return metrics_epoch, other_data_epoch
    