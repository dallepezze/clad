from re import X
import numpy as np
import os 
import pickle

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset,Subset
from torchvision import transforms

from PIL import Image
import PIL

from src.utilities import utility_logging
import imgaug.augmenters as iaa
import cv2
from src.models.draem_add.perlin import rand_perlin_2d_np
import glob
from argparse import Namespace

normalazition_parameters_mvtec = {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)}

transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

normalize_transforms = {  "mvtec": transforms.Normalize(**normalazition_parameters_mvtec)   }
MVTEC_CLASS_NAMES = ['hazelnut', 'bottle', 'cable','capsule', 'metal_nut', 'pill', 'toothbrush', 'transistor', 'zipper', 'screw']
labels_datasets = { "mvtec":MVTEC_CLASS_NAMES  }


'''transformation of the images'''
def create_transform_img(img_size, crp_size):
    transform = []
    if crp_size != img_size:
        transform.append(transforms.Resize(crp_size, Image.ANTIALIAS))
        #transform.append(transforms.Resize((scale_size,scale_size), Image.ANTIALIAS)) #you can try this line for transforming the images from memory
        #transform.append(transforms.CenterCrop(crp_size))
    else:
        transform.append(transforms.Resize((img_size,img_size), Image.BICUBIC))

    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))
    return transforms.Compose(transform)    

def filter_dataset(dataset, task_id, batch_size): 
    """
    This methods extract from the entire dataset in input the data corresponding to specific task(s)

    Args:
        dataset (torch.Dataset) : dataset to be considered
        task_id (int) : id of the task
        batch_size (int) : batch size
    
    Return:
        class_subset (torch.Subset) : subset of the dataset
        class_dataloader (torch.DataLoader) : dataloader over the subset
    """
    if np.asarray(task_id).ndim==0:
        class_idx = np.where((dataset.targets==task_id))[0]
    else:
        class_idx = np.where( np.isin(dataset.targets,task_id) )[0]

    class_subset = Subset(dataset, class_idx)

    class_loader = DataLoader(class_subset, shuffle=True, batch_size=batch_size, drop_last=True)
    #class_loader = DataLoader(class_subset, shuffle=True, batch_size=batch_size)

    return class_subset,class_loader

def create_transform_x(opt,crp_size): #prepares images as inputs for backbone
    transform = []
    if opt.img_size == opt.crp_size:
        transform.append(transforms.Resize((opt.img_size, opt.img_size), interpolation=2))
    else: 
        transform.append(transforms.Resize(opt.img_size, Image.ANTIALIAS))
        transform.append(transforms.CenterCrop(crp_size))

    if opt.gray == True:
        gray_transform = transforms.Grayscale(num_output_channels=3)
        transform.append(gray_transform)  
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))
    return transforms.Compose(transform)    

def create_transform_x_with_rotation(opt,crp_size,rotation_degree,fill=0):#the same as above, but with rotation
    transform = []
    transform.append(transforms.Resize((crp_size, crp_size), interpolation=2))
    random_rotation = torchvision.transforms.RandomAffine(degrees=rotation_degree,resample=PIL.Image.BICUBIC,fill=fill)
    transform.append( random_rotation )
    if opt.gray == True:
      gray_transform = transforms.Grayscale(num_output_channels=3)
      transform.append(gray_transform)
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(**normalazition_parameters_mvtec))
    return transforms.Compose(transform)

def load_dataset(parameters,type_dataset,normalize=True):
    """
    Function that returns the dataset split into training and test

    Args:
        parameters (dict) : json with all the execution parameters
        type_dataset (str) : name of the dataset ('mvtec' or 'shangai' for the moment only mvtec is supported)
        normalize (bool) : boolean that states if we must perform normalization on the data

    Return:
        dataset_train (MVTecDataset) : training MVTec Dataset
        dataset_test (MVTecDataset) : test MVTec Dataset
    """
    
    print(f"Type of Dataset: {type_dataset}")

    dataset_name = type_dataset.lower()
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list+= [normalize_transforms[type_dataset]]
    train_transform = transforms.Compose(transform_list)    

    filepath = f"{parameters['data_path']}{dataset_name}"#not relevant
    print(f"filepath dataset: {filepath}")#not relevant

    if type_dataset=="mvtec":
        opt = utility_logging.from_parameters_to_opt(parameters)  
        apply_rotation=parameters.get("apply_rotation",False)
        opt.apply_rotation=apply_rotation
        dataset_train = MVTecDataset(opt, is_train=True)#entire train dataset
        opt.apply_rotation=False
        dataset_test = MVTecDataset(opt,is_train=False)#entire test dataset
    else:
        raise ValueError(f"{type_dataset} dataset is not present !")

    return dataset_train, dataset_test


'''
Import mvtec dataset

├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   ├── broken_small
│   │   └── contamination
│   ├── test
│   │   ├── broken_large
│   │   ├── broken_small
│   │   ├── contamination
│   │   └── good
│   └── train
│       └── good
...

'''
        

class MVTecDataset(Dataset):
    '''
    Import MVTecDataset with all the classes containing objects  
    '''
    def __init__(self, opt,is_train=True):
        assert opt.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(opt.class_name, MVTEC_CLASS_NAMES)
        self.opt = opt
        self.dataset_path = opt.data_path
        self.class_name = opt.class_name
        self.is_train = is_train
        self.cropsize = opt.crp_size
        self.use_all_classes = opt.use_all_classes#True
        self.only_normal = opt.only_normal
        self.only_anomalies = opt.only_anomalies
        self.apply_rotation = opt.apply_rotation
        self.architecture = opt.architecture
        #added for draem
        self.anomaly_source_paths = opt.anomaly_source_paths

        if self.only_normal is True and self.only_anomalies is True:
            raise ValueError("only_normal=True and only_anomalies=True")

        if self.use_all_classes:
            self.x, self.y, self.anomaly_info, self.mask, self.filepaths = self.load_dataset_folders()
        else:
            self.x, self.y, self.anomaly_info, self.mask, self.filepaths = self.load_dataset_folder()
        '''  
        # mask
        if self.cropsize != self.opt.img_size:
            self.transform_mask = transforms.Compose([transforms.Resize(self.opt.img_size, Image.NEAREST), transforms.CenterCrop(self.cropsize), transforms.ToTensor()])
        else:
            self.transform_mask = transforms.Compose([transforms.Resize(self.opt.img_size, Image.NEAREST), transforms.ToTensor()])
        '''  
        rgb_dict = { "white":(255,255,255),"blue":(80,116,151),"brown":(164,115,93),"black":(0,0,0),"gray":(190,190,190)  }
        colors = ["white","blue","gray","black","black","black","gray","black","brown","white"]
        rgb_colors = [ rgb_dict[color_class] for color_class in colors ]
        self.rgb_colors = rgb_colors

        new_x,new_y,new_anomaly_info,new_mask,new_filepaths = [],[],[],[],[]
        
        self.y = np.asarray(self.y)
        self.targets = np.asarray(self.y)#dataset.targets
        self.anomaly_info = np.asarray(self.anomaly_info)
        self.loaded_from_memory = False

        #self.anomaly_source_paths = ""
        #self.augmenters = []
        #self.rot = None

        #draem
        if self.architecture == "draem":
            self.resize_shape = [256, 256]
            #self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/*/*.jpg"))
            self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                iaa.pillike.EnhanceSharpness(),
                iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                iaa.Solarize(0.5, threshold=(32,128)),
                iaa.Posterize(),
                iaa.Invert(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.Affine(rotate=(-45, 45))
            ]

            self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    #for DRAEM
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image_draem_test(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return image

    def transform_image_draem(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7 #30% chance the original image to be rotated
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):#fcn built in MVTecDataset class to open the image from filepath and extract its info
        x, y,anomaly_info, mask,filepath = self.x[idx], self.y[idx],self.anomaly_info[idx], self.mask[idx], self.filepaths[idx]
        
        if self.architecture != "draem":
            try:
                #x = Image.open(x)
                x = Image.open(x).convert('RGB')
            except:
                print(f"I am not able to load image at path: {x}")

            crp_size = self.cropsize
            if self.apply_rotation:
                rotation_degree = self.rotation_degrees[idx]
                fill = self.rgb_colors[y]
                transform_img = create_transform_x_with_rotation(self.opt,crp_size,rotation_degree,fill)     
            else:
                transform_img = create_transform_x(self.opt,crp_size)
            

        class_name = MVTEC_CLASS_NAMES[y]
        '''
        if self.cropsize == self.opt.img_size:
            if class_name in ['zipper', 'screw', 'grid'] and self.loaded_from_memory is False:  # handle greyscale classes
                x = np.expand_dims(np.array(x), axis=2)
                x = np.concatenate([x, x, x], axis=2)
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        '''
        
        '''
        if anomaly_info == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        '''  
        
        if self.architecture == 'efficientad':  
            x = torch.cat([torch.unsqueeze(transform_img(x),dim=0), torch.unsqueeze(transform_img(transform_ae(x)),dim=0)])
            return x, y, idx, anomaly_info, filepath
        elif self.architecture == 'draem':
            #print(self.anomaly_source_paths)
            if self.is_train == True:
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                image, augmented_image, anomaly_mask, has_anomaly = self.transform_image_draem(x,
                                                                                self.anomaly_source_paths[anomaly_source_idx])
                sample = {'image': torch.from_numpy(image), "anomaly_mask": torch.from_numpy(anomaly_mask),
                        'augmented_image': torch.from_numpy(augmented_image), 'has_anomaly': torch.from_numpy(has_anomaly)}
                return sample, y, idx, anomaly_info, filepath
            else:
                return torch.from_numpy(self.transform_image_draem_test(x)), y, idx, anomaly_info, filepath
        else:
            x = transform_img(x)
            return x, y, idx, anomaly_info, filepath
        
        
    def get_wrapper(self,idx):
        x, y, idx, anomaly_info, filepath  = self.__getitem__(idx)
        mask = self.mask[idx]
        diz = {"x":x, "y":y, "idx":idx, "anomaly_info":anomaly_info,"mask":mask, 
              "filepath":filepath, "real_A":x}
        from argparse import Namespace
        ns = Namespace(**diz)
        return ns

    def get_mask(self,mask_path,anomaly_info):
        if self.cropsize != self.opt.img_size:
            self.transform_mask = transforms.Compose([transforms.Resize(self.opt.img_size, Image.NEAREST), transforms.CenterCrop(self.cropsize), transforms.ToTensor()])
        else:
            self.transform_mask = transforms.Compose([transforms.Resize(self.opt.img_size, Image.NEAREST), transforms.ToTensor()])

        if anomaly_info == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask_path)
            mask = self.transform_mask(mask)
        return mask 


    def __len__(self):
        return len(self.x)

    def load_dataset_folders(self): #exclusively when one needs to load all classes
        lista_x,lista_y,lista_anomaly_info,lista_mask,lista_filepaths = [],[],[],[],[]
        for class_name in MVTEC_CLASS_NAMES: #class_name is string while class_idx ranges from 0 to 9 
            self.class_name = class_name
            x, y, anomaly_info,mask,filepaths = self.load_dataset_folder()
            lista_x.extend(x)
            lista_y.extend(y)
            lista_anomaly_info.extend(anomaly_info)
            lista_mask.extend(mask) 
            lista_filepaths.extend(filepaths)
        return lista_x,lista_y,lista_anomaly_info,lista_mask,lista_filepaths


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, anomaly_info, y, mask,filepaths = [], [], [], [],[]

        index = np.where( np.asarray(MVTEC_CLASS_NAMES)==self.class_name)[0][0]
        index_class = index

        img_dir = os.path.join(self.dataset_path, self.class_name, phase).replace('\\','/')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth').replace('\\','/')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type).replace('\\','/')
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f).replace('\\','/')
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            # load gt labels
            if img_type == 'good':
                if self.only_anomalies is False:
                    anomaly_info.extend([0] * len(img_fpath_list)) #it was y in CFA {0,1}
                    mask.extend([None] * len(img_fpath_list))
                    y.extend([index_class] * len(img_fpath_list)) # if it is toothbrush then y = 7
                    x.extend(img_fpath_list) # x is filepath just as in CFA
                    filepaths.extend(img_fpath_list) # same as x
            else:
                if self.only_normal is False:
                    anomaly_info.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(gt_dir, img_type).replace('\\','/')
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png').replace('\\','/')
                                    for img_fname in img_fname_list]
                    mask.extend(gt_fpath_list)
                    y.extend([index_class] * len(img_fpath_list))
                    x.extend(img_fpath_list)
                    filepaths.extend(img_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(anomaly_info), list(mask), list(filepaths) 


class MemoryDataset(Dataset):
    def __init__(self, filepaths, strategy): #, dataset_current_task
    
        self.strategy = strategy

        self.filepaths = filepaths

        filepaths_dict = [ filepath_dict for filepath_dict,filepath_img in filepaths]#filepath_dict,filepath_img: list of .pickles and .pngs for samples stored in memory 
        #corresponding to one class only, that will be replayed in CL setting for current task
        indices_original,filepaths_original,class_ids = [],[],[]
        for filepath_dict in filepaths_dict:
            f = open(filepath_dict, "rb")
            diz = pickle.load(f)
            f.close()

            y, idx, anomaly_info, filepath = diz["y"],diz["idx"],diz["anomaly_info"],diz["filepath"]
            indices_original.append(idx)
            filepaths_original.append(filepath)
            class_ids.append(y)
        self.indices_original = np.asarray(indices_original)
        self.filepaths_original = np.asarray(filepaths_original)
        self.class_ids = np.asarray(class_ids)

        self.anomaly_source_paths = self.strategy.parameters['anomaly_source_paths']

        if self.strategy.parameters['architecture'] == "draem":
            self.resize_shape = [256, 256]
            #self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/*/*.jpg"))
            self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                iaa.pillike.EnhanceSharpness(),
                iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                iaa.Solarize(0.5, threshold=(32,128)),
                iaa.Posterize(),
                iaa.Invert(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.Affine(rotate=(-45, 45))
                ]

            self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.filepaths)
    

    #for DRAEM
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)


    def transform_image_draem(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7 #30% chance the original image to be rotated
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly



    def __getitem__(self, idx):#get item by using idx under which it's saved 
        filepath_dict,filepath_img = self.filepaths[idx] #filepaths = list of tuples (pickle, img_path)

        f = open(filepath_dict, "rb")
        diz = pickle.load(f)
        f.close()

        y, idx, anomaly_info, filepath = diz["y"],diz["idx"],diz["anomaly_info"],diz["filepath"]
        class_id = y

        if self.strategy.parameters["architecture"] != "draem":
            #img = Image.open(filepath_img)
            img = Image.open(filepath_img).convert('RGB')
        
        img_size = self.strategy.parameters['img_size']   
        crp_size = self.strategy.parameters['crp_size']          
        transform_img = create_transform_img(img_size, crp_size)

        if self.strategy.parameters['architecture'] == 'efficientad':  
            x = torch.cat([torch.unsqueeze(transform_img(img),dim=0), torch.unsqueeze(transform_img(transform_ae(img)),dim=0)])
            return x, np.asarray(y), np.asarray(idx), np.asarray(anomaly_info), filepath
        elif self.strategy.parameters['architecture'] == 'draem':  
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image_draem(filepath_img,
                                                                            self.anomaly_source_paths[anomaly_source_idx])
            sample = {'image': torch.from_numpy(image), "anomaly_mask": torch.from_numpy(anomaly_mask),
                    'augmented_image': torch.from_numpy(augmented_image), 'has_anomaly': torch.from_numpy(has_anomaly)}
            return sample, y, idx, anomaly_info, filepath
        else:
            x = transform_img(img)
            return x, np.asarray(y), np.asarray(idx), np.asarray(anomaly_info), filepath

        
class ContinualLearningBenchmark:#returns task_stream = (train, test dataset) in specified order by the array task_order
    def __init__(self,complete_train_dataset, complete_test_dataset, num_tasks, task_order):
        self.complete_train_dataset = complete_train_dataset
        self.complete_test_dataset = complete_test_dataset
        self.num_tasks = num_tasks
        self.task_order = task_order

        if num_tasks!=len(task_order):
            print("Attention! Number of tasks!=task_order length")

    def produce_task_stream(self):
        datasets_train_list = []
        datasets_test_list = []

        for task_id in self.task_order:
            dataset_train_task,_ = filter_dataset(self.complete_train_dataset, task_id, 1)
            dataset_test_task,_ = filter_dataset(self.complete_test_dataset, task_id, 1)
            datasets_train_list.append(dataset_train_task)
            datasets_test_list.append(dataset_test_task)

        train_stream = datasets_train_list
        test_stream  = datasets_test_list
        
        task_stream = train_stream,test_stream

        return task_stream