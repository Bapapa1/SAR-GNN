import torch
import os
from .utils import accuracy,reduce_mean,data_loader
from GNN_models.GCN import GCN_model

from .EarlyStopper import Patience
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch import distributed as dist
from pathlib import Path
from thop import profile

def apply_model(args, dataset, config,train_graph_id, test_graph_id,best_epoch=1000
                ,exp_class='hyper_select',config_id=0):

    shuffle = config.pop('shuffle')
    lr = config.pop('learning_rate')
    weight_decay = config.pop('weight_decay')
    epochs = config.pop('epochs')
    early = config.pop('early_stopper')
    if exp_class=='outer_model':
        epochs=best_epoch+1
    else:
        early_stopper = Patience(patience=early['patience'], use_loss=early['use_loss'])  # only for val_dataset

    batch_size = config.pop('batch_size')
    device=args.device

    train_loader, test_loader= data_loader(dataset,train_graph_id, test_graph_id,batch_size, shuffle,exp_class)
    kwargs = config
    '''------------------------------------------'''
    if args.GNN_models=='GCN':
        model=GCN_model(args, **kwargs)

    else:
        raise EnvironmentError


    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fun = nn.CrossEntropyLoss()
    exp = experience(device, model, optimizer, lr, loss_fun)

    for epoch in range(epochs):

        train_acc,train_loss= exp.train_model(train_loader)
        test_acc, test_loss = exp.assess_model(test_loader)

        if exp_class == 'hyper_select':
            if early_stopper.stop(epoch, test_loss, test_acc,train_loss, train_acc):
                if args.local_rank==0:
                    print(f'----Stopping at epoch {epoch}, best is {early_stopper.get_best_vl_metrics()}----')
                break
        if epoch % args.log_every == 0 or epoch == 1 or epoch==(epochs-1):
            if exp_class == 'hyper_select':
                print(
                    'Epoch: {},TR loss: {:.3f},TR acc: {:6.3f}% , VL loss: {:.3f} ,VL acc: {:6.3f}%'.format(epoch,train_loss,train_acc*100,test_loss,
                                                                                                           test_acc * 100))
            else:
                print(
                    'Epoch: {},TR loss: {:.3f},TR acc: {:6.3f}% ,TE loss: {:.3f} ,TE acc: {:6.3f}%'.format(epoch,
                                                                                                           train_loss,
                                                                                                           train_acc * 100,
                                                                                                           test_loss,
                                                                                                           test_acc * 100))
        scheduler.step()

    if exp_class == 'hyper_select':
        train_loss, train_acc, val_loss, val_acc, best_epoch = early_stopper.get_best_vl_metrics()


    return train_loss, train_acc,test_loss, test_acc,best_epoch



class experience():
    def __init__(self, device, model, optimizer,lr,loss_fun):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.lr_de=lr
        self.loss_fun=loss_fun


    def train_model(self, train_loader):

        loss_all = 0
        acc_all = 0
        total=0.0
        acc_all_num=0

        model = self.model
        optimizer = self.optimizer
        model.train()
        # print(len(train_loader))
        for i,dataset in enumerate(train_loader):
            dataset.to(self.device)
            output= model(dataset)
            graph_labels = dataset.y
            loss= self.loss_fun(output,graph_labels,)
            acc,acc_rate=accuracy(output,graph_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_graphs = dataset.num_graphs

            loss_all+=loss.item()*num_graphs
            total += graph_labels.size(0)
            acc_all+=acc_rate.item()*num_graphs
            acc_all_num += acc.item()

        return acc_all / total, loss_all / total

    def assess_model(self, data_loader):

        model = self.model
        #model.eval()

        loss_all = 0
        acc_all = 0
        total=0
        acc_all_num=0

        with torch.no_grad():

            for i,dataset in enumerate(data_loader):
                dataset.to(self.device)
                graph_labels = dataset.y
                output= model(dataset)
                loss = self.loss_fun(output, graph_labels, )
                acc,acc_rate = accuracy(output, graph_labels)
                num_graphs = dataset.num_graphs
                loss_all += loss.item() * num_graphs
                total += graph_labels.size(0)
                acc_all += acc_rate.item() * num_graphs
                acc_all_num+= acc.item()

            return acc_all/total, loss_all / total












