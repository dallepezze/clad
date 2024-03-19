import torch
from src.models.cfa_add.metric import *

#Added
import sys
from torch import tensor
from typing import Tuple


def create_padim(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    padim = PaDiM(device, parameters['d_reduced'], parameters['backbone_name'])
    return padim, device  


class KNNExtractor(torch.nn.Module):
	def __init__(
		self, device,
		backbone_name : str = "wide_resnet50_2",
		out_indices : Tuple = None,
		pool_last : bool = False,
	):
		super().__init__()

		self.device = device

		'''self.feature_extractor = timm.create_model(
			backbone_name,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)
		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		
		self.feature_extractor.eval()
		self.feature_extractor = self.feature_extractor.to(self.device)'''
                
		self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
		self.backbone_name = backbone_name # for results metadata
		self.out_indices = out_indices
			
	def __call__(self, x: tensor, model):
		model.eval()
		with torch.no_grad():
			feature_maps = model(x)
		feature_maps = [fmap.detach().cpu() for fmap in feature_maps]
		if self.pool:
			# spit into fmaps and z
			return feature_maps[:-1], self.pool(feature_maps[-1])
		else:
			return feature_maps

class PaDiM(KNNExtractor):
	def __init__(
		self, device,
		d_reduced: int = 350,
		backbone_name: str = "wide_resnet50_2",
	):
		super().__init__(device,
			backbone_name=backbone_name,
			out_indices=(1,2,3),
		)
		self.image_size = 224
		self.d_reduced = d_reduced
		self.epsilon = 0.04 # cov regularization
		self.resize = None


