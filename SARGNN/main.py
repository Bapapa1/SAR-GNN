import torch
import argparse
from datetime import timedelta
import time
import pyhocon
from models.utils import dataset_error
from models.data_splits import load_data
from K_fold.hyper_select import select
from K_fold.outer_assess import outerAssessment
from torch import distributed as dist
from K_fold.grid import grid_search
import numpy as np
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='parameter of RD')

    parser.add_argument('--cuda', default=True, help='cuda training')
    parser.add_argument('--seed', default=4, help='random seed')
    parser.add_argument('--data_config', type=str, default='data_setting.conf')
    parser.add_argument('--detail', default=True, help='show the train settings')
    parser.add_argument('--log_every', default=10,help='show training result every 10 epochs')
    parser.add_argument('--heads', default=8,help='only for GAT heads')
    parser.add_argument('--outer_folds', default=10,help='cross-validation')
    parser.add_argument('--Integration_method', default='Multiplication', help='Multiplication（Scaling regularizatio） or Addition (Weighted sum)')
    parser.add_argument('--GNN_models', default='GCN', help='GCN')
    parser.add_argument('--repeat', default=3, help='train times in outer-fold')
    parser.add_argument('--dataset_name', default='MUTAG')

    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training') #参数’–local_rank’也不需要手动给出,会在下面通过命令行里辅助工具torch.distributed.launch自动给出

    return parser.parse_args()



def show_args(args,):
    argsDict = args.__dict__
    if args.local_rank == 0:
        print('------the models settings are as following--------')
        for key in argsDict:
            print(key, ':', argsDict[key])


def model_device():
    if args.cuda and args.local_rank == 0:
        print('-------USE CUDA-----------')
        device_id = torch.cuda.current_device()  # 返回当前所选设备的索引
        print('num GPU:', torch.cuda.device_count())

        # print('Device:', device, device_id, torch.cuda.get_device_name(device_id))
    elif not args.cuda:
        print('-------USE CPU-----------')
        print('Device:', device)


def format_time(time):
    time = timedelta(seconds=time)
    total_seconds = int(time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    fo = f'{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(time.microseconds)}'
    return fo


if __name__ == '__main__':
    args = get_args()
    args.cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device
    model_device()
    '''torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed) '''

    args.data_root = f'{args.GNN_models}_DATASET_{args.Integration_method}'
    if args.Integration_method=='Addition':
        args.hyper_file= Path('hyper_config')/f'{args.GNN_models}_hyper_model.yml'
    elif args.Integration_method=='Multiplication':
        args.hyper_file= Path('hyper_config')/f'{args.GNN_models}_mutl_hyper_model.yml'
    else:
        raise IOError


    '''load data'''
    train_test, train_val, val_set, test_set, dataset = \
        load_data(args=args)

    args.input_features = dataset.num_features
    args.num_classes=dataset.num_classes
    if args.detail:
        show_args(args)

    '''Creating a hyperparametric search grid'''
    config = grid_search(args.dataset_name, args.hyper_file, args)
    config = config.hyper_list  # list(dict)

    '''Search for optimal hyperparameters in the grid'''
    start = time.time()
    model_select = select(args,train_val,val_set)
    best_config = model_select.hyper_select(dataset,config)
    select_end = time.time()


    '''Training the dataset with optimal hyperparameters'''
    train_model = outerAssessment(args, best_config,train_test, test_set)
    run = train_model.assess(dataset)
    if args.local_rank == 0:
        print('\n')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!training finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    train_end = time.time()

    interval_select = format_time(select_end - start)
    interval_train = format_time(train_end - select_end)
    sum_time = format_time(train_end - start)
    if args.local_rank == 0:
        print(f'interval_select : {interval_select}, interval_test : {interval_train},'
          f'sum_time : {sum_time}')




