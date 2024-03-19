import os 
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Tuple, cast, Optional
import torch
import cv2
from skimage.segmentation import find_boundaries


    
def denorm(image):   
    #print(f"test img shape: {image.shape}")     
    normalazition_parameters_mvtec = {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)}
    image = np.transpose(image, (1,2,0))
    image = ((image * normalazition_parameters_mvtec["std"]) + normalazition_parameters_mvtec["mean"]) * 255
    image = image.astype(np.uint8)
    return image
#returns image in range [0,255]


def denorm_draem(image):        
    image = np.transpose(image, (1,2,0))* 255
    image = image.astype(np.uint8)
    return image
#returns image in range [0,255]

def plot_predict(self, defect_class, map_combined, gt_mask_list, indices, threshold, test_imgs):

    map_combined = map_combined.copy()
    map_combined = map_combined[:, np.newaxis, :, :]
    #print(map_combined.shape)

    for i in range(map_combined.shape[0]):
        if not os.path.exists(os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}').replace('\\','/')):
            os.makedirs(os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}').replace('\\','/'))

        file_binary = os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}/img_out_{indices[i]}.png').replace('\\','/')    

        fig, axes = plt.subplots(1, 5, figsize = (20,4))
        original_mask = gt_mask_list[i]
        colormap_image = plt.get_cmap('jet')(map_combined[i,0])
        binary_mask = np.where(map_combined[i] > threshold, 1, 0)
        bounded = boundary_image(denorm(test_imgs[i]), binary_mask[0])
        #file = os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}/img_{indices[i]}.png').replace('\\','/')
        #plt.imsave(file, colormap_image)
       

        # Display the images on the subplots
        axes[0].imshow(denorm(test_imgs[i]))
        axes[0].axis('off')
        axes[0].set_title('Original image', fontsize=12, pad=10, loc='center') 

        axes[1].imshow(original_mask[0], cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('GroundTruth', fontsize=12, pad=10, loc='center') 

        axes[2].imshow(colormap_image, cmap='jet')
        axes[2].axis('off')
        axes[2].set_title('Predicted heat map', fontsize=12, pad=10, loc='center') 

        axes[3].imshow(binary_mask[0], cmap='gray')
        axes[3].axis('off')
        axes[3].set_title('Predicted mask', fontsize=12, pad=10, loc='center')  # Add title below the image

        axes[4].imshow(bounded)
        axes[4].axis('off')
        axes[4].set_title('Segmentation result', fontsize=12, pad=10, loc='center') 

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(file_binary, bbox_inches='tight')
        plt.close()



def plot_predict_draem(self, defect_class, map_combined, gt_mask_list, indices, threshold, test_imgs):

    map_combined = map_combined.copy()
    #print(f'Map to be saved: {map_combined.shape}')
    map_combined = map_combined[:, np.newaxis, :, :]
    #print(map_combined.shape)

    for i in range(map_combined.shape[0]):
        if not os.path.exists(os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}').replace('\\','/')):
            os.makedirs(os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}').replace('\\','/'))

        file_binary = os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}/img_out_{indices[i]}.png').replace('\\','/')    

        fig, axes = plt.subplots(1, 5, figsize = (20,4))
        original_mask = gt_mask_list[i]
        colormap_image = plt.get_cmap('jet')(map_combined[i,0])
        binary_mask = np.where(map_combined[i] > threshold, 1, 0)
        bounded = boundary_image(denorm_draem(test_imgs[i]), binary_mask[0])
        #file = os.path.join(self.strategy.test_output_dir, f'{defect_class[i]}/img_{indices[i]}.png').replace('\\','/')
        #plt.imsave(file, colormap_image)
       

        # Display the images on the subplots
        axes[0].imshow(denorm_draem(test_imgs[i]))
        axes[0].axis('off')
        axes[0].set_title('Original image', fontsize=12, pad=10, loc='center') 

        axes[1].imshow(original_mask[0], cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('GroundTruth', fontsize=12, pad=10, loc='center') 

        axes[2].imshow(colormap_image, cmap='jet')
        axes[2].axis('off')
        axes[2].set_title('Predicted heat map', fontsize=12, pad=10, loc='center') 

        axes[3].imshow(binary_mask[0], cmap='gray')
        axes[3].axis('off')
        axes[3].set_title('Predicted mask', fontsize=12, pad=10, loc='center')  # Add title below the image

        axes[4].imshow(bounded)
        axes[4].axis('off')
        axes[4].set_title('Segmentation result', fontsize=12, pad=10, loc='center') 

        plt.subplots_adjust(wspace=0.4)
        plt.savefig(file_binary, bbox_inches='tight')
        plt.close()







def boundary_image(image: Union[np.ndarray, torch.Tensor],
                   patch_classification: Union[np.ndarray, torch.Tensor],
                   boundary_color: Tuple[int, int, int] = (255, 0, 0)
                   ) -> np.ndarray:
    """
       Draw boundaries around masked areas on image.

       Args:
           image: Image on which to draw boundaries.
           patch_classification: Mask defining the areas.
           boundary_color: Color of boundaries.

       Returns:
           b_image: Image with boundaries.

    """

    image = to_numpy(image).copy()
    mask = to_numpy(patch_classification).copy()

    found_boundaries = find_boundaries(mask).astype(np.uint8)#mask is (256,256)
    layer_two = np.zeros(image.shape, dtype=np.uint8)
    
    layer_two[:, :, 0] = boundary_color[0]  # Assign red channel
    layer_two[:, :, 1] = boundary_color[1]  # Assign green channel
    layer_two[:, :, 2] = boundary_color[2]  # Assign blue channel

    b_image = composite_image(image, layer_two, found_boundaries)#{(3,256,256),(3,256,256),(256,256)}

    return b_image



#helping functions
def composite_image(image_one: Union[np.ndarray, torch.Tensor],
            image_two: Union[np.ndarray, torch.Tensor],
            mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Draws image_two over image_one using a mask,
    Areas marked a
s 1 is transparent and 0 draws image_two with
    opacity 1.

    Args:
        image_one: The base image.
        image_two: The image to draw with.
        mask: mask on where to draw the image.

    Returns:
        tot_Image: The combined image.

    """
    image_one = to_numpy(image_one).copy()
    image_two = to_numpy(image_two).copy()
    mask = to_numpy(mask).copy()

    height, width, channels = image_one.shape

    image_two = cv2.resize(image_two, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

    mask = (mask == (1 | True))
    image_one[mask] = image_two[mask]

    return image_one
#output value (3,256,256)


def to_numpy(in_array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Casts a tensor to a numpy array

    Args:
        in_array: The array to be casted

    Returns:
        np_array: a casted numpy array

    """
    if isinstance(in_array, torch.Tensor):
        in_array = in_array.cpu().numpy()

    np_array = cast(np.ndarray, in_array)

    return np_array