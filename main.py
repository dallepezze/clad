import argparse
import os
import sys
import shutil
from datetime import datetime
import warnings
#ADDED
import time
import os

ROOT = ".."
sys.path.append(ROOT)
sys.path.append(ROOT+"/pytorch_pix2pix")

#from src.models.fastflow import *
from src.models.cfa import *
from src.strategy_ad import *
#from src.trainer.trainer_fastflow import *
from src.trainer.trainer_cfa import *
from src.datasets import *
from src.utilities.utility_main import *
from src.utilities import utility_logging
from src.utilities.utility_models import *
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Parser to take filepaths")
parser.add_argument("--parameters_path", type=str, nargs="?", action = 'store', help="parameters path", default="test_cfa_ideal_replay.json" )       #loads specific parameters from .json file for specific model

#ONLY FOR DEBUG
#parser.add_argument("--credentials_path", type=str, nargs="?", action = 'store', help="credentials path", default="credentials.json")                #load the credentials for wandb logging
#parser.add_argument("--default_path", type=str, nargs="?", action = 'store', help="default parameters path", default="common_param.json")            #common parameters for the training

parser.add_argument("--seed", type=int, nargs="?", action = 'store', help="seed", default=random.randint(1,10000))                                    #set seed

#load paths to the parameters variable (model_specific, credentials for Neptune, common params)
args = parser.parse_args()
path = 'configurations'
parameters_path = os.path.join(path,args.parameters_path).replace('\\','/')
credentials_path = os.path.join(path,"credentials.json").replace('\\','/')
default_path = os.path.join(path,"common_param.json").replace('\\','/')
seed = args.seed

print(f"seed: {seed}")
print(f"parametes_path: {parameters_path}")
print(f"credentials_path: {credentials_path}")
print(f"default_path: {default_path}")

#seed = 43

#Get wandb run object,parameters and available device
run, parameters, device = init_execute(credentials_path, default_path, parameters_path, seed)
project_name = run.project
experiment_name = run.id

now = datetime.now() # current date and time
date_time = now.strftime("%d_%m_%Y__%H-%M-%S")
path_logs = os.path.join(f"logs/{project_name}/{experiment_name}_{date_time}").replace('\\','/')#Neptune path for logging data
print(f"path_logs: {path_logs}")
utility_logging.create_paths([path_logs]) 

filename = os.path.basename(parameters_path)
dst = os.path.join(path_logs,filename).replace('\\','/')#/logs/{project_name}/{experiment_name}_{date_time}/test_fast_flow_standard.json
shutil.copyfile(parameters_path, dst)                   #copy parameters (specific for the model) to Neptune

# Load Dataset
channels,dataset_name,num_tasks,task_order = parameters["channels"],parameters["dataset_name"],parameters["num_tasks"],parameters["task_order"]
complete_train_dataset, complete_test_dataset,train_stream,test_stream = load_and_split_dataset(parameters,dataset_name,num_tasks,task_order)#returns train and test dataset in default task order, and then in specified order for CL
#output values: MVTecDataset (it contains list of x,y...), MVTecDataset(same), list(of Subsets), list(of Subsets)

labels_map = create_new_labels_map(labels_datasets[dataset_name], task_order, num_tasks) #put strings of classes' names in desired task order
print(f"labels_map: {labels_map}")

# Create Strategy
if isinstance(complete_train_dataset[0][0], dict):
    input_size = complete_train_dataset[0][0]["image"].shape#input_size=256
else:
    input_size = complete_train_dataset[0][0].shape#input_size=256
print(f"input_size: {input_size}")

original_stdout = sys.stdout # Save a reference to the original standard output
filepath = os.path.join(path_logs, 'model_info.txt').replace('\\','/') #cretes model_info.txt file within the created project on Neptune

with open( filepath, 'w') as f:
    sys.stdout = f #change output default destination
    strategy = create_strategy(parameters,run,labels_map,device,path_logs,input_size)#creates strategy.trainer,.test_loss_function, .input_size, .device
    ''' self.parameters = parameters

        self.num_tasks = num_tasks
        self.task_order = task_order
        self.num_epochs = num_epochs
        self.labels_map = labels_map
        self.path_logs = path_logs
        self.run = run'''
sys.stdout = original_stdout 

import copy
original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream = copy.deepcopy(complete_train_dataset),copy.deepcopy(complete_test_dataset),copy.deepcopy(train_stream),copy.deepcopy(test_stream)
complete_train_dataset,complete_test_dataset,train_stream,test_stream = manage_dataset(strategy, parameters,complete_train_dataset,complete_test_dataset,train_stream,test_stream)
num_tasks = strategy.num_tasks
elapsed_time = 0
init_strategy_variables(strategy, complete_train_dataset,complete_test_dataset,train_stream,test_stream,original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream,labels_map,run,path_logs,elapsed_time)

sample_strategy = strategy.parameters.get("sample_strategy")
test_only_seen_tasks = strategy.parameters.get("test_only_seen_tasks")

if sample_strategy=="multi_task" and test_only_seen_tasks:
    raise ValueError("test_only_seen_tasks is True but you are in multi_task mode")


for index_training in range(0,num_tasks):#0...9
    train_dataset = train_stream[index_training]
    test_dataset = test_stream[index_training]

    strategy.index_training = index_training
    strategy.train_task_id = task_order[index_training]#class in task_order on specific place (on index_training)
    strategy.task_label = labels_map[index_training]#name of the class
    task_label = strategy.task_label

    print(f"\nStart Training Task T{index_training} ({ task_label })")
    
    current_train_dataset,current_test_dataset = strategy.init_variables_dataset(train_dataset,test_dataset )
    #    it does the following:
    #    self.task_train_dataset = task_train_dataset
    #    self.task_test_dataset = task_test_dataset

    #    self.current_train_dataset = current_train_dataset
    #    self.current_test_dataset = current_test_dataset

    # LOAD Memory 
    # assign memory to strategy and load it from memory(use_memory) or create a new one(new_memory)    use_memory,memory_dataset_path_train,memory_dataset_path_test,type_memory_train,type_memory_test,memory_model_path,new_memory,sample_strategy = give_memory_parameters(strategy.parameters)
    load_memory_main(strategy, strategy.parameters["memory_dataset_path_train"], strategy.parameters["type_memory_train"])#(strategy, "", "memorized")

    #Load MemoryReconstruct
    load_memory_reconstruct_main(strategy, "train", strategy.parameters["memory_reconstruct_dataset_path_train"], strategy.parameters["type_memory_reconstruct_train"])
    load_memory_reconstruct_main(strategy, "test" , strategy.parameters["memory_reconstruct_dataset_path_test"], strategy.parameters["type_memory_reconstruct_test"])

    if strategy.sample_strategy=="single_model":
        print("Reset Trainer")
        reset_trainer(strategy)

    # LOAD MODEL (if memory_model_path!="")
    load_model_main(strategy)#it does not work until we specify use_memory = True

    # SAVE MODEL
    if index_training==0:
        save_model_main(strategy)#saved on Wandb
        

    # TRAINING
    print(f"\nTraining Task T{index_training}")
    batch_size = strategy.batch_size 
    num_epochs = strategy.num_epochs 
    strategy.training_task(current_train_dataset,current_test_dataset,num_epochs,batch_size)

    memory_update_main(strategy)

    # SAVE MODEL
    save_model_main(strategy)
    
    print(f"Training time: {strategy.elapsed_time} seconds")
    # EVALUATION
    print("\nEvaluation:")
    strategy.evaluate_test_stream(test_stream, batch_size=8)
    plt.close("all")


run.log({"Training Time": strategy.elapsed_time})
print(f"Training time: {strategy.elapsed_time} seconds")
#run["Finished"].log(True)
run.finish()




