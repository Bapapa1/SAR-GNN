
from torch_geometric.datasets import TUDataset
from torch_geometric import transforms
import torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import json
import torch.distributed as dist
from models.social_degree import social_deg

def load_data(args):
    data_name=args.dataset_name
    data_root = args.data_root
    if data_name in ['MUTAG','ENZYMES']:
        dataset = TUDataset(root=f'./{data_root}', name=data_name, use_node_attr= True)
    elif data_name in ['IMDB-BINARY','IMDB-MULTI']:
        dataset = TUDataset(root=f'./{data_root}', name=data_name,transform=social_deg(cat=True))
        '''for Social networks dataset: add degree as node feature'''
    else:
        raise

    label_list=dataset.data.y
    num_graphs=len(dataset)
    '''Split the dataset'''
    train_test,train_val,val_set,test_set=dataset_spilts(args,num_graphs,label_list,data_name,data_root)

    return train_test,train_val,val_set,test_set,dataset


def dataset_spilts(args,idx,y,data_name,file_root):
        #K-fold
        name=f'./{file_root}/{data_name}/{data_name}_splits.json'
        print('------start dataset splits process--------')
        splits = []
        graph_index=range(idx)

        train_test=[]
        train_val=[]
        val_set=[]
        test_set=[]

        outer_k=StratifiedKFold(n_splits=args.outer_folds,shuffle=True,random_state=args.seed)
        #outer_k=StratifiedKFold(n_splits=self.K_train_test,shuffle=True)

        for outer_train,outer_test in outer_k.split(X=graph_index,y=y):

            train_test.append(outer_train)
            test_set.append(outer_test)

            outer_train_labels = y[outer_train]


            split={'test':outer_test,'model_select':[]}

            val_s=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=args.seed)
            for i,j in val_s.split(outer_train,outer_train_labels):
                inter_train=outer_train[i]
                inter_val =outer_train[j]
                train_val.append(inter_train)
                val_set.append(inter_val)

            split['model_select'].append({'train':inter_train,'val':inter_val})

            splits.append(split)


        print('--------dataset splits finished---------')
        return train_test,train_val,val_set,test_set








