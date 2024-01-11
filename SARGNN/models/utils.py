import numpy as np
import torch
import torch.utils.data as Data

import torch.nn.functional as F
import json
import scipy.sparse as sp
from collections import Counter
from torch_geometric.data import DataLoader
from torch import distributed as dist

def dataset_error(args,set_data):
    data_name=args.dataset_name
    datasets=set_data['data_name']
    classify=None
    for name,i in datasets.items():
        if data_name in i:
            classify=name

    return classify



def data_loader(dataset,train_graph_id, test_graph_id, batch_size,shuffle,exp_class):

    train_dataset=dataset[train_graph_id.tolist()]
    test_dataset=dataset[test_graph_id.tolist()]

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    if exp_class == 'hyper_select':
        test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    elif exp_class == 'outer_model':
        test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader,test_loader


def explicit_broadcast(this, other):
    # Append singleton dimensions until this.dim() == other.dim()
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)

    # Explicitly expand so that shapes are the same
    return this.expand_as(other)

def accuracy(output, labels):
    #_,a=torch.max(output,1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct,correct/labels.size(0)



def normalize(adj):
    """Row-normalize tensor matrix"""
    rowsum = np.array(adj.sum(1))  # 对每一行求和
    r_inv = np.power(rowsum, -0.5).flatten()  # flatten()降到一维数组
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv)
    return adj.dot(r_mat_inv).transpose().dot(r_mat_inv).tocoo()


def reduce_mean(tensor, nprocs):  # 用于所有gpu上的运行结果，比如loss
    with torch.no_grad():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt/=nprocs

    '''dist.all_reduce方法会在所有GPU设备上求得该值的和，然后将该值除以world_size就得到了该值在所有设备上的均值了。
    注意，这里对多个设备上的loss求平均不是为了backward，仅仅是查看做个记录。这里有很多人误认为，在使用多GPU时需要先求平均损失然后在反向传播，
    其实不是的。应该是每个GPU设备计算出各批次数据的损失后，通过backward方法计算出各参数的损失梯度，然后DDP会自动帮我们在多个GPU设备上对各参数的损失梯度求平均，
    最后通过optimizer.step()去更新各参数'''

    return rt

def reduce_sum(tensor, nprocs):
    with torch.no_grad():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    return rt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

