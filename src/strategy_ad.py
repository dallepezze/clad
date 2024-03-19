import numpy as np 
import gc
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.metrics import *
from src.utilities import utility_logging
from src.datasets import *
from src.memory import *
from src.utilities.utility_pix2pix import produce_input_scaling_model

# Models
from src.models.vae import *
from src.models.cae import *
from src.models.pix2pix_model import *
from src.models.fastflow import *


# Trainers
from src.trainer.trainer_vae import *
from src.trainer.trainer_cae import *
from src.trainer.trainer_pix2pix_inpaint import *
from src.trainer.trainer_pix2pix_scaling_model import *
from src.trainer.trainer_pix2pix_inpaint import Trainer_Pix2Pix_InPaint
from src.trainer.trainer_fastflow import Trainer_fastflow
from src.trainer.trainer_fastflow import *
from src.trainer.trainer_vae import *
from src.trainer.trainer_pix2pix import Trainer_Pix2Pix

#Added
from src.models.cfa import *
from src.trainer.trainer_cfa import *
from src.models.cfa_add.cnn.resnet import wide_resnet50_2 as wrn50_2

#Subsequently added for EfficientAD
from adcl_paper.src.models.efficientad import *
from adcl_paper.src.trainer.trainer_efficientad import *
from src.models.st import *
from src.trainer.trainer_st import *
import os
from tqdm import tqdm

#Added
from adcl_paper.src.models.patchcore import *
from adcl_paper.src.trainer.trainer_patchcore import *
import timm

#Added
from src.models.padim import *
from src.trainer.trainer_padim import *


#Added
from src.utilities.utility_ad import forgetting


#Added
from src.models.draem import *
from src.trainer.trainer_draem import *

#Added
from adcl_paper.src.models.stfpm import *
from adcl_paper.src.trainer.trainer_stpfm import *

#ADDED
import time

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Attributes:
       param patience: how many epochs to wait before stopping when loss is not improving
       param min_delta: minimum difference between new loss and old loss for new loss to be considered as an improvement
     
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True          
 
def create_trainer(strategy,parameters,device,input_size,lr,batch_size):
    num_tasks, latent_dim, num_epochs, batch_size, lr, task_order, sample_strategy, beta, dataset_name= utility_logging.return_values_from_parameters(parameters)

    print(f"input_size: {input_size}")
    criterion_type = parameters["criterion_type"]
    print(f"criterion_type: {criterion_type}")
    loss_function = create_loss_function(parameters,criterion_type,input_size, beta)

    from argparse import Namespace
    opt = utility_logging.from_parameters_to_opt(parameters)  
    
    # ARCHITECTURE
    if "architecture" in parameters: 
        architecture = parameters["architecture"]
    else: 
        architecture = "vae"

    print(architecture)

    if architecture=="vae":
        vae, device = create_vae_model(strategy,input_size, parameters)
        latent_dim = parameters['latent_dim']
        trainer = Trainer_VAE(strategy,vae, batch_size, latent_dim, device)
    elif architecture == "cae":
        cae, device = create_cae_model(strategy,input_size,parameters)
        trainer = Trainer_CAE(strategy,input_size, cae, device,latent_dim)
    elif architecture == "fastflow":
        fastflow, device = create_fastflow(input_size, parameters)
        trainer = Trainer_fastflow(strategy,fastflow)

        #Added
    elif architecture == "cfa":
        cfa, device = create_cfa(strategy,input_size, parameters)
        trainer = Trainer_cfa(strategy,cfa)   

        #Added
    elif architecture == "patchcore":
        patch, device = create_patchcore(strategy,input_size, parameters)
        trainer = Trainer_patchcore(strategy,patch)   

        #Added
    elif architecture == "padim":
        padim, device = create_padim(strategy,input_size, parameters)
        trainer = Trainer_padim(strategy,padim)   

        #Added
    elif architecture == "draem":
        draem, device = create_draem(strategy,input_size, parameters)
        trainer = Trainer_draem(strategy,draem)   

        #Added
    elif architecture == "stfpm":
        storig, device = create_stfpm(strategy,input_size, parameters)
        trainer = Trainer_STFPM(strategy,storig)   

        #Added
    elif architecture == "efficientad":
        if parameters["st"] == True:
            st, device = create_st(strategy,input_size, parameters)
            trainer = Trainer_st(strategy,st)  
        else:
            eff, device = create_efficientad(strategy,input_size, parameters)
            trainer = Trainer_efficientad(strategy,eff)  

    elif architecture=="pix2pix":
        trainer_param = parameters["trainer"]
        if trainer_param=="pix2pix_inpaint":
            pix2pix_vae,opt,encoder,decoder = create_pix2pix_model(parameters["device_id"])
            trainer = Trainer_Pix2Pix_InPaint(strategy,pix2pix_vae,opt)
        elif trainer_param=="pix2pix_scaling_model":
            pix2pix_vae,opt,encoder,decoder = create_pix2pix_model(parameters["device_id"])
            trainer = Trainer_Pix2Pix(strategy,pix2pix_vae,opt, produce_input_scaling_model)
        else:
            raise ValueError(f"trainer {architecture} not found !")
    else:
        raise ValueError(f"trainer {architecture} not found !")

    trainer.device = device
    trainer.batch_size = batch_size

    return trainer

def reset_trainer(strategy):
    if strategy.parameters['architecture'] == 'cfa':
        strategy.trainer.ad_model.C = None
        strategy.trainer.ad_model.D = None
        strategy.trainer.ad_model.Descriptor = None
        strategy.trainer.optimizer = None
        strategy.trainer.ad_model.r = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()

    if strategy.parameters['architecture'] == 'efficientad':
        strategy.trainer.ad_model.teacher = None
        strategy.trainer.ad_model.student = None
        if strategy.parameters['st'] == False:
            strategy.trainer.ad_model.autoencoder = None
        strategy.trainer.optimizer = None
        strategy.trainer.scheduler = None
        strategy.trainer.ad_model.teacher_mean = None
        strategy.trainer.ad_model.teacher_std = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()

    if strategy.parameters['architecture'] == 'patch':
        for i in range (len(strategy.trainer.ad_model.list_mem)):
            strategy.trainer.ad_model.list_mem[i] = None
        #strategy.trainer.vae.feature_extractor = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()


    if strategy.parameters['architecture'] == 'padim':
        for i in range (len(strategy.trainer.MEAN)):
            strategy.trainer.MEAN[i] = None
            strategy.trainer.COV[i] = None
            strategy.trainer.r_indices = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()

    if strategy.parameters['architecture'] == 'draem':
        strategy.trainer.ad_model.model = None
        strategy.trainer.ad_model.model_seg = None
        strategy.trainer.optimizer = None
        #added
        strategy.trainer.scheduler = None
        
        strategy.trainer.ad_model.loss_l2 = None
        strategy.trainer.ad_model.loss_ssim = None
        strategy.trainer.ad_model.loss_focal = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Trainer is reset")

    if strategy.parameters['architecture'] == 'stfpm':
        strategy.trainer.ad_model.teacher = None
        strategy.trainer.ad_model.student = None
        strategy.trainer.optimizer = None
        strategy.trainer.ad_model = None
        gc.collect()
        torch.cuda.empty_cache()


    trainer = create_trainer(strategy,strategy.parameters,strategy.device,strategy.input_size,strategy.parameters["lr"],strategy.parameters["batch_size"])
    strategy.trainer = trainer

def create_strategy(parameters,run,labels_map,device,path_logs,input_size):
    '''
    Returns the strategy corresponding to a specific model architecture.
    '''
    num_tasks, latent_dim, num_epochs, batch_size, lr, task_order, sample_strategy, beta, dataset_name= utility_logging.return_values_from_parameters(parameters)

    print(f"input_size: {input_size}")
    criterion_type = parameters["criterion_type"]
    print(f"criterion_type: {criterion_type}")
    loss_function = create_loss_function(parameters,criterion_type,input_size, beta)

    from argparse import Namespace
    opt = utility_logging.from_parameters_to_opt(parameters)    

    # CREATE STRATEGY    
    strategy = Strategy_CL_AD(parameters,None, num_tasks, task_order, num_epochs, labels_map, path_logs, run)
    strategy.device = device
    
    if strategy.parameters['architecture'] == 'cfa':
        strategy.model = wrn50_2(pretrained=True, progress=True)
        strategy.model = strategy.model.to(strategy.device)
        strategy.model.eval()

    if strategy.parameters['architecture'] == 'patchcore':
        strategy.model = timm.create_model("wide_resnet50_2",out_indices=(2,3),features_only=True,pretrained=True)
        for param in strategy.model.parameters():
            param.requires_grad = False
        strategy.model = strategy.model.to(strategy.device)
        strategy.model.eval()

    if strategy.parameters['architecture'] == 'padim':
        strategy.model = timm.create_model("wide_resnet50_2",out_indices=(1,2,3),features_only=True,pretrained=True)
        for param in strategy.model.parameters():
            param.requires_grad = False
        strategy.model = strategy.model.to(strategy.device)
        strategy.model.eval()  

    trainer = create_trainer(strategy,parameters,device,input_size,lr,batch_size)
    
    strategy.trainer = trainer

    # other variables
    strategy.test_loss_function = create_mse_function(input_size)
    strategy.input_size = input_size


    return strategy

class Strategy_CL_AD: 
    '''
    It trains and evaluate the model on a Continual Learning setting.

    Attributes:
       parameters: execution parameters
       trainer: contains the model, lr,batch_size,...
       num_tasks(int): number of tasks - Ex: num_tasks=10
       task_order(list): The order of classes to be executed from task 0 to task 9 - Ex: task_order=[0,1,..,9]
       num_epochs(int)
       labels_map(dict): list of classes' names
       path_logs(string): path where to save logs
       run: wandb run to track the experiments

    '''
    def __init__(self, parameters, trainer, num_tasks, task_order, num_epochs, labels_map, path_logs, run):

        self.trainer = trainer

        vars(self).update(parameters)
        self.parameters = parameters

        self.num_tasks = num_tasks
        self.task_order = task_order
        self.num_epochs = num_epochs
        self.labels_map = labels_map
        self.path_logs = path_logs
        self.run = run


    def init_variables_dataset(self,train_dataset,test_dataset ):
        '''
        Parameters:
          train_dataset: task train dataset
          test_dataset: task test dataset

        Returns:
          train and test datasets corresponding to the current task
        '''
        print(f"Length of current dataset: {len(train_dataset)}")
        task_train_dataset = train_dataset
        task_test_dataset = test_dataset
        
        current_train_dataset = task_train_dataset
        current_test_dataset = task_test_dataset

        self.task_train_dataset = task_train_dataset
        self.task_test_dataset = task_test_dataset

        self.current_train_dataset = current_train_dataset
        self.current_test_dataset = current_test_dataset

        #if self.parameters['architecture'] == 'eff':
        if self.index_training == 9 or self.parameters['architecture'] in ['efficientad','storig']:
            self.train_output_dir = os.path.join(self.parameters["output_dir"], 'trainings',
                                        MVTEC_CLASS_NAMES[self.index_training]).replace('\\','/')
            self.test_output_dir = os.path.join(self.parameters["output_dir"], 'anomaly_maps',
                                    MVTEC_CLASS_NAMES[self.index_training], 'test').replace('\\','/')
            os.makedirs(self.train_output_dir, exist_ok=True)
            os.makedirs(self.test_output_dir, exist_ok=True)

        self.checkpoints = os.path.join(self.parameters["output_dir"], 'checkpoints').replace('\\','/')
        os.makedirs(self.checkpoints, exist_ok=True)

        return current_train_dataset,current_test_dataset


    def return_strategy_parameters(self):
        return self.index_training,self.train_task_id,self.task_label,self.labels_map,self.task_order,self.num_tasks,self.run,self.path_logs


    def update_state(self,metrics,other_data,metrics_epoch,other_data_epoch, mode="train"): # mod="train" or "val" or "test"
        '''
        It saves the evaluation metrics task on Wandb and print them. Metrics are obtained during each training epoch in training_task fcn. They are calcualted both in train
        and eval mode.

        '''
        task_index = self.index_training #ranges [0,9]
        epoch_index = self.current_epoch
        epoch = epoch_index
        num_epochs = self.num_epochs
        metrics[task_index].setdefault(epoch_index, {} ) 
        other_data[task_index].setdefault(epoch_index, {} ) 
        
        metrics_epoch = {  f"{mode}_"+key:metrics_epoch[key] for key in metrics_epoch }
        other_data_epoch = {f"{mode}_"+key:other_data_epoch[key] for key in other_data_epoch  if "x_hats" not in key }

        metrics[task_index][epoch_index].update(metrics_epoch)
        other_data[task_index][epoch_index].update( other_data_epoch )

        for key in metrics_epoch:
            if isinstance(metrics_epoch[key], np.ndarray) or isinstance(metrics_epoch[key], list):
                for value in list(metrics_epoch[key]):
                    if value==np.nan or value==np.inf or value==-np.inf:
                        value = sys.maxint
                    self.run.log({f"Task_Results/T{self.index_training}/epochs/{mode}/epoch_{epoch}/{key}":value})
            else:
                #self.run[f"Task_Results/T{self.index_training}/epochs/{mode}"][key].log(metrics_epoch[key])
                self.run.log({f"Task_Results/T{self.index_training}/epochs/{mode}/{key}": metrics_epoch[key]})
        
        dict_print = { key:metrics_epoch[key] for key in metrics_epoch if np.asarray(metrics_epoch[key]).ndim==0 }

        if mode=="train":
            print(f'\n ====> Epoch {epoch + 1}/{num_epochs} \t Train Results: {dict_print} \t ')
        elif mode=="val":
            print(f'\n ====> Epoch {epoch + 1}/{num_epochs} \t Validation Results: {dict_print} \t ')


    def training_task(self,current_train_dataset,current_test_dataset,num_epochs,batch_size,eval=True):
        """
        It trains the model on the train dataset
        """ 
        #ADDED
        start_time = time.time()
        if self.parameters["architecture"] == "storig" :
            current_train_data_loader = DataLoader(current_train_dataset , batch_size=batch_size, pin_memory      =          True,
                                    shuffle         =          True)  
        else:
            current_train_data_loader = DataLoader(current_train_dataset , batch_size=batch_size, pin_memory      =          True,
                                    shuffle         =          True,
                                    drop_last       =          True)
        if self.parameters["architecture"] in ["patchcore", "padim", "draem"]:
            current_test_data_loader = DataLoader(current_test_dataset , batch_size = 1, pin_memory      =          True)
        else:
            current_test_data_loader = DataLoader(current_test_dataset , batch_size=batch_size, pin_memory      =          True)
        '''
        if self.parameters["architecture"] == "eff":
            current_train_data_loader = InfiniteDataloader(current_train_data_loader)
            current_test_data_loader = InfiniteDataloader(current_test_data_loader)
        '''
        self.current_train_data_loader = current_train_data_loader
        self.current_test_data_loader = current_test_data_loader
        
        if self.parameters["architecture"] == "efficientad":
            print("Device teacher: " + str(self.trainer.ad_model.device))
            self.trainer.ad_model.teacher.eval()
            self.trainer.ad_model.teacher_mean, self.trainer.ad_model.teacher_std = teacher_normalization(self.trainer.ad_model.teacher, current_train_data_loader)

        index_training, train_task_id, task_label,labels_map,task_order,num_tasks,run,path_logs = self.return_strategy_parameters()

        save_model_param = self.parameters.get("save_model", True)
        if self.lr_scheduler:
            print("Lr scheduler used")
            if self.parameters["architecture"] == "cfa":
                self.trainer.optimizer  = torch.optim.AdamW(params        = self.trainer.ad_model.parameters(), 
                                    lr            = self.lr,
                                    weight_decay  = 5e-4,
                                    amsgrad       = True )
            else:
                self.trainer.optimizer = torch.optim.Adam(self.trainer.ad_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            scheduler = ReduceLROnPlateau(self.trainer.optimizer, mode='min', patience=3, min_lr=1e-6, factor=0.5)

        # Early Stopping
        if self.early_stopping:
            print("Early Stopping used")
            early_stopping = EarlyStopping(patience=self.parameters['patience'])
        

        if self.parameters["architecture"] == "cfa":
            self.trainer.ad_model._update_centroid(current_train_data_loader,self.model)

        #added    
        self.early_stop = False
        self.best_epoch = 0

        epoch=0
        losses = []
        n_critic_eval = self.n_critic_eval 
        
        #ADDED
        end_time = time.time()
        add_elapsed_time = end_time - start_time
        self.elapsed_time = self.elapsed_time+add_elapsed_time
        while(epoch<num_epochs): 
            #ADDED
            start_time = time.time()
            self.current_epoch = epoch
            # TRAIN EPOCH
            print(f"current_epoch: {epoch}")
            self.mode = "train"
            metrics_epoch,other_data_epoch  = self.trainer.train_epoch(current_train_data_loader)#self.trainer.train_epoch!
            losses.append(metrics_epoch['loss'])
            self.update_state(self.metrics_train,self.other_data_train,metrics_epoch,other_data_epoch, mode="train")
            #ADDED
            end_time = time.time()
            add_elapsed_time = end_time - start_time
            self.elapsed_time = self.elapsed_time+add_elapsed_time
            # TEST EPOCH
            if self.parameters["architecture"] not in ["patchcore", "padim"]:
                if eval==True and self.current_epoch%n_critic_eval==0:
                    self.mode = "test"
                    metrics_epoch,other_data_epoch = self.trainer.test_epoch(current_test_data_loader)#self.trainer.test_epoch!
                    self.update_state(self.metrics_test,self.other_data_test,metrics_epoch,other_data_epoch, mode="val")
                    if self.lr_scheduler:
                        #scheduler.step(metrics_epoch['loss'])
                        scheduler.step(metrics_epoch['f1'])
                    if self.early_stopping:
                        #early_stopping(metrics_epoch['loss'])
                        early_stopping(1-metrics_epoch['f1'])
                        if early_stopping.early_stop:
                            self.best_epoch = epoch - early_stopping.patience
                            self.early_stop = True
                            break

            epoch+=1        


    def evaluate_test_stream(self, test_stream, batch_size):
        '''
        It evaluates performance of model on the test dataloader
        '''
        index_training, train_task_id, task_label,labels_map,task_order,num_tasks,run,path_logs = self.return_strategy_parameters()
        from src.utilities.utility_main import give_memory_parameters,give_ad_parameters
        use_memory,memory_dataset_path_train,memory_dataset_path_test,type_memory_train,type_memory_test,memory_model_path,new_memory,sample_strategy = give_memory_parameters(self.parameters)
        anomaly_detection_task,anomaly_detection_task_with_metrics,test_only_seen_tasks,num_tasks_to_examine = give_ad_parameters(self.parameters,index_training)
        
        #loading in case of early stopping EFF
        if self.early_stop == True and self.parameters['architecture'] == "efficientad" and self.parameters['st'] == False:
            run_name1 = 'model_student'+str(self.best_epoch)
            run_name2 = 'model_autoencoder'+str(self.best_epoch)
            #self.trainer.ad_model.model.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl"), map_location='cuda:0'))
            self.trainer.ad_model.student.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl")))
            #self.trainer.ad_model.model_seg.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name2 + ".pckl"), map_location='cuda:0'))
            self.trainer.ad_model.autoencoder.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name2 + ".pckl")))

            self.trainer.ad_model.student.eval()
            self.trainer.ad_model.autoencoder.eval()
            
        #loading in case of early stopping ST 
        if self.early_stop == True and self.parameters['architecture'] == "efficientad" and self.parameters['st'] == True:
            run_name1 = 'model_student'+str(self.best_epoch)
            #self.trainer.ad_model.model.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl"), map_location='cuda:0'))
            self.trainer.ad_model.student.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl")))

            self.trainer.ad_model.student.eval()

        if self.early_stop == True and self.parameters['architecture'] == "storig":
            run_name1 = 'model_student'+str(self.best_epoch)
            #self.trainer.ad_model.model.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl"), map_location='cuda:0'))
            self.trainer.ad_model.student.load_state_dict(torch.load(os.path.join(self.checkpoints, run_name1 + ".pckl")))

            self.trainer.ad_model.student.eval()


        diz_test,other_data,lista_metriche  = {}, {}, {}
        for test_task_index in range(num_tasks_to_examine): 
            if self.test_all_dataset_together:
                test_task_index = index_training

            self.current_test_task_index = test_task_index
            test_task_index = self.current_test_task_index
            test_task_id = task_order[test_task_index]
            self.test_task_id = test_task_id
            class_name = self.labels_map[test_task_index]
            architecture_name = self.parameters["architecture"]
            trainer_name = self.parameters.get("trainer", "")

            diz_test[test_task_index], other_data[test_task_index] = {}, {}
            test_dataset = test_stream[test_task_index]
            if architecture_name == "patchcore" or architecture_name == "padim" or architecture_name == "draem" or architecture_name == "efficientad":
                test_data_loader = DataLoader(test_dataset , batch_size = 1, pin_memory = True)
            else:
                test_data_loader = DataLoader(test_dataset , batch_size=batch_size, pin_memory = True)

            self.mode = "evaluate_data"
            if anomaly_detection_task and anomaly_detection_task_with_metrics:
                #addde "cfa"
                if architecture_name not in ["fastflow", "cfa", "efficientad", "patchcore", "padim", "draem","storig"] and trainer_name!="pix2pix_inpaint" and trainer_name!="classification":
                    print("reconstruct_epoch_with_evaluation_ad")
                    diz = reconstruct_epoch_with_evaluation_ad(self, self.parameters, test_data_loader, self.complete_test_dataset,class_name,self.index_training,test_task_index,self.run,self.path_logs)
                    metrics_epoch, other_data_epoch = diz, {}
                else:
                    print("Evaluate_data")
                    metrics_epoch,other_data_epoch = self.trainer.evaluate_data(test_data_loader,self.test_loss_function)
            if anomaly_detection_task and anomaly_detection_task_with_metrics is False:
                print("reconstruct_epoch")
                diz = reconstruct_epoch(self, self.parameters, test_data_loader, self.complete_test_dataset,class_name,self.index_training,test_task_index,self.run,self.path_logs)
                metrics_epoch, other_data_epoch = diz, {}
            if anomaly_detection_task is False and anomaly_detection_task_with_metrics is False:
                print("Evaluate_data")
                metrics_epoch,other_data_epoch = self.trainer.evaluate_data(test_data_loader,self.test_loss_function)

            diz_test[test_task_index].update(metrics_epoch)
            other_data[test_task_index].update(other_data_epoch)

            # Update AD metrics
            if anomaly_detection_task and anomaly_detection_task_with_metrics:
                diz_metriche = metrics_epoch
                for key in diz_metriche:
                    lista_metriche.setdefault(key, [])
                    lista_metriche[key].append(diz_metriche[key])
                    #run[f"Task_Results/T{index_training}/evaluation_ad/{key}"].log(diz_metriche[key])
                    run.log({f"Task_Results/T{index_training}/evaluation_ad/{key}": diz_metriche[key]})

            plt.close("all")

        self.metrics_test[self.index_training].update(diz_test)
        self.other_data_test[self.index_training].update(other_data)

        if anomaly_detection_task and anomaly_detection_task_with_metrics:
            for key in lista_metriche:
                save_summary_value(self,run,lista_metriche,index_training,key)

        for test_task_index in range(num_tasks_to_examine):   
            if self.test_all_dataset_together:
                loss = self.metrics_test[self.index_training][index_training]["loss"]
            else:
                loss = self.metrics_test[self.index_training][test_task_index]["loss"]
            #run[f"Task_Results/T{self.index_training}"]["loss"].log(loss)
            run.log({f"Task_Results/T{self.index_training}/loss": loss})
            print(f'     Test loss TASK T{test_task_index} during training T{index_training}: {loss}')


        forget = forgetting(self.metrics_test, self.index_training)
        run.log({f"Task_average_forgetting/T{self.index_training}": forget})


        return diz_test, other_data


def save_summary_value(strategy,run,lista_metriche,index_training,key):
    sample_strategy = strategy.parameters.get("sample_strategy")
    if sample_strategy!="multi_task" and sample_strategy!="single_model":
        value = np.asarray(lista_metriche[key])[0:index_training+1].mean()
        #run[f"Summary/evaluation_ad/{key}"].log(value)
        run.log({f"Summary/evaluation_ad/{key}": value})
    else:
        if sample_strategy=="multi_task":
            value = np.asarray(lista_metriche[key])[0:].mean()
            for i in range(10):
                #run[f"Summary/evaluation_ad/{key}"].log(value)
                run.log({f"Summary/evaluation_ad/{key}": value})

        if sample_strategy=="single_model":
            value = np.asarray(lista_metriche[key])[index_training]
            average_value = value
            for i in range(index_training):
                path_metric = f"Task_Results/T{i}/evaluation_ad/{key}"
                #values = np.asarray(run[path_metric].fetch_values()["value"].values)
                #values = np.asarray(metric_values = run.history(keys=[path_metric])["value"])
                #values = np.asarray(metric_values = run.summary[path_metric])
                #metric_values = run.summary[path_metric]
                #values = np.asarray(metric_values)
                #value = values[i]
                #value = np.asarray(lista_metriche[key])[i]
                #print(values.shape)
                #value = values
                #average_value = average_value + value
            #average_value = average_value/(index_training+1)
            #run[f"Summary/evaluation_ad/{key}"].log(average_value)
            #run.log({f"Summary/evaluation_ad/{key}": average_value})
            run.log({f"Summary/evaluation_ad/{key}": value})


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    teacher.eval()

    mean_outputs = []
    for batch in tqdm(train_loader, desc='Computing mean of features'):
        if torch.cuda.is_available():
            train_image = batch[0][:,0].to("cuda:0")
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for batch in tqdm(train_loader, desc='Computing std of features'):
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

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)