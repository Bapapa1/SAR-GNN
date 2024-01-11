
import json
import os
import numpy as np
from pathlib import Path
from models.utils import NumpyEncoder
from models.Experiment import apply_model
import copy
import torch

class outerAssessment():
    def __init__(self,args, best_config,train_test, test_set):
        self.dataset_name = args.dataset_name
        self.best_config=best_config
        self.root = Path(args.data_root) / args.dataset_name / 'models'
        self.result_root = self.root / 'assessment_result.json'
        if not self.root.exists():
            os.mkdir(self.root)
        self.args = args
        self.R = args.repeat  # 重复次数
        self.K_fold = args.outer_folds

        self.train=train_test
        self.test=test_set


    def assess(self, dataset):

        train_graph_id = self.train
        test_graph_id = self.test


        test_acc_list = []
        avg_test_acc = 0.0
        train_acc_list = []
        avg_train_acc = 0.0
        sum_result = {}


        for outer_K in range(self.K_fold):
            if self.args.local_rank == 0:
                print('\n')
                print(f'--------------------This step is {outer_K}-th outer assessment----------------------- ')
            test_id = test_graph_id[outer_K]
            train_id = train_graph_id[outer_K]

            configs=self.best_config[f'{outer_K}_outer']

            best_epoch=configs['best_epoch']
            best_config=configs['best_config']


            result = {}
            train_acc_all = 0.0
            test_acc_all = 0.0

            if self.args.local_rank == 0:
                print('the best hyper-param of {}_fold is :best_epoch :{}, best_val_acc :{:6.3f}%,best_train_acc :{:6.3f}%'
                  .format(outer_K,best_epoch,configs['best_val_acc']*100,configs['best_train_acc']*100),'\n',
                  'best_config:{}'.format(configs['best_config'])
                  )
                print('\n')

            for th in range(1, self.R + 1):
                if self.args.local_rank == 0:
                    print(f'-------------run {th}th repeat------------')
                config = copy.deepcopy(best_config)

                '''------------------------------------------------------------------------------------------'''
                train_loss, train_acc, test_loss, test_acc ,_= apply_model(self.args, dataset, config,
                                                                               train_id, test_id,
                                                                               best_epoch,
                                                                               exp_class='outer_model',config_id=th)
                '''------------------------------------------------------------------------------------------'''

                train_acc_all += train_acc
                test_acc_all += test_acc

                result[f'{th}'] = {'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc,
                                   'test_loss': test_loss}
                if self.args.local_rank == 0:
                    print('{} results : train_acc:{:6.3f}%,train_loss:{:.3f} , test_acc:{:6.3f}%,test_loss:{:.3f}'
                      .format(th,train_acc*100,train_loss,test_acc*100,test_loss))

            mean_test_acc = test_acc_all / self.R
            mean_train_acc = train_acc_all / self.R

            torch.cuda.empty_cache()

            test_acc_list.append(mean_test_acc)
            train_acc_list.append(mean_train_acc)

            avg_test_acc += mean_test_acc
            avg_train_acc += mean_train_acc

            sum_result[f'{outer_K}'] = {'repeat': result, 'mean_test_acc': mean_test_acc,
                                        'mean_train_acc': mean_train_acc}
            if self.args.local_rank == 0:
                print('\n')
                print('{}_outer sum result is : mean_test_acc:{:6.3f}%,mean_train_acc:{:6.3f}%'
                  .format(outer_K,mean_test_acc*100,mean_train_acc*100))

        avg_test_acc = avg_test_acc / self.K_fold
        avg_train_acc = avg_train_acc / self.K_fold
        std = np.std(test_acc_list)

        sum_result['summary'] = {'train_outer': avg_train_acc, 'test_outer': avg_test_acc,'test_std':std,
                                 'train_acc_list': train_acc_list, 'test_acc_list': test_acc_list}

        with open(self.result_root, 'w') as f:
            json.dump(sum_result, f, cls=NumpyEncoder)

        if self.args.local_rank == 0:

            print('\n')
            print('----------{} result is : avg_test_acc= {:6.3f}%,avg_train_acc={:6.3f}%, std={:6.3f}%--------------'
                  .format(self.dataset_name,avg_test_acc*100,avg_train_acc*100,std*100))
