import torch
import numpy as np
from tqdm import tqdm
from src.models.cfa_add.metric import *

#Added
import sys
from torchvision import transforms
from torch import tensor
from sklearn import random_projection
from typing import Tuple
#import timm
from PIL import ImageFilter


TQDM_PARAMS = {
	"file" : sys.stdout,
	"bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}

def get_tqdm_params():
    return TQDM_PARAMS

def get_coreset_idx_randomp(
    z_lib : tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
) -> tensor:
    """Returns n coreset idx for given z_lib.
    
    Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
    CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)

    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.

    Returns:
        coreset indices
    """

    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        #added for "cl" case
        z_lib = z_lib.cpu()
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print( "   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)
    # The line below is not faster than linalg.norm, although i'm keeping it in for
    # future reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n-1), **TQDM_PARAMS):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances) # iterative step
        select_idx = torch.argmax(min_distances) # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)


def create_patchcore(strategy, img_shape, parameters):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    patch = PatchCore(device, parameters['f_coreset'], parameters['backbone_name'], parameters['coreset_eps'])
    return patch, device  


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

class PatchCore(KNNExtractor):
	def __init__(self, device, 
		f_coreset: float = 0.1, # fraction the number of training samples
		backbone_name : str = "wide_resnet50_2",
		coreset_eps: float = 0.90, # sparse projection parameter
	):
		super().__init__(device,
			backbone_name=backbone_name,
			out_indices=(2,3),
		)
		self.f_coreset = f_coreset
		self.coreset_eps = coreset_eps
		self.image_size = 224
		self.average = torch.nn.AvgPool2d(3, stride=1)
		self.n_reweight = 3
		self.blur = GaussianBlur(4)
		self.list_mem = []
		self.resize = None
        
        
        
class GaussianBlur:
    def __init__(self, radius : int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0]/map_max).filter(self.blur_kernel)
        )*map_max
        return final_map


