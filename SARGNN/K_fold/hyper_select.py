
import json
import os
import torch
from pathlib import Path
from models.utils import NumpyEncoder
from K_fold.grid import grid_search
from models.Experiment import apply_model
import copy
import time


class select():
    def __init__(self, args,train_val,val_set):
        self.dataset_name = args.dataset_name
        self.hyper_file = args.hyper_file
        self.device = args.device
        self.log_every = args.log_every
        self.args = args
        self.root = Path(args.data_root) / args.dataset_name / 'models' / 'hyper'
        if not self.root.exists():
            os.makedirs(self.root)
        self.train_val=train_val
        self.val_set=val_set

    def hyper_select(self, dataset,config):  # 主运行函数

        if self.args.local_rank == 0:
            print('\n')
            print('-----------------------------------run best-hyper select-----------------------------------')


        train_graph_id = self.train_val
        val_graph_id = self.val_set
        K_fold = self.args.outer_folds
        best_config_root = self.root / 'outer_best_config.json'
        best_config = {}
        for outer_K in range(K_fold):
            config_list = copy.deepcopy(config)
            train_id=train_graph_id[outer_K]
            val_id=val_graph_id[outer_K]
            if self.args.local_rank == 0:
                print('\n')
                print(f'---------------------now {outer_K}th hyper-param select---------------------')

            best_config[f'{outer_K}_outer']= self._val_select(config_list, outer_K,dataset,train_id,val_id)

            torch.cuda.empty_cache()
        if self.args.local_rank == 0:
            print('-----------------------------------select hyper-param is finish-----------------------------------')


        return best_config



    def _val_select(self,config_list, outer_K,dataset,train_id,val_id):

        best_epoch=1000
        best_config_id = 0
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_val_loss = 0.0


        config_result_root = self.root / f'{outer_K}_outer_config_result.json'
        results={}


        for i, config in enumerate(config_list):
            config_copy=copy.deepcopy(config)
            if self.args.local_rank == 0:
                print(f'----this is {i + 1} config experience----')

            train_loss, train_acc, val_loss, val_acc,epoch  = apply_model(self.args, dataset, config,
                                                                         train_id, val_id, exp_class='hyper_select',config_id=i)

            result = {'train_acc': train_acc, 'val_acc': val_acc,
                       'train_loss':train_loss,'val_loss':val_loss,'config':config_copy,'epoch':epoch}

            results[f'{i}_config']=result
            torch.cuda.empty_cache()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_epoch=epoch
                best_config=config_copy
                best_config_id=i



        best_result = {'best_config_id': best_config_id, 'best_val_acc': best_val_acc, 'best_val_loss': best_val_loss,
                       'best_train_acc': best_train_acc, 'best_train_loss': best_train_loss,'best_epoch':best_epoch ,
                       'best_config': best_config}

        if self.args.local_rank == 0:
            print('the best hyper-param of {}_fold is :config_id:{},best_epoch :{},best_val_acc :{:6.3f}%,best_train_acc :{:6.3f}%'.format(outer_K,best_config_id, best_epoch,best_val_acc * 100,
                      best_train_acc * 100),'\n'
                'best_config:{}'.format(best_config))

        with open( config_result_root, 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)

        return best_result








