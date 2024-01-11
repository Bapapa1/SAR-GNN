from torch.nn import BatchNorm1d
import torch
from models.M_layers import M_cross_layers
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_geometric.utils import add_self_loops, degree
from torch import einsum
from .gcn_con import GCNConv

class GCN_model(nn.Module):
    def __init__(self,args, dropout,GNN_layers, cross_MLP_layers, X_dim, M_dim, sim_mutl_method,inner_dim):
        super(GCN_model, self).__init__()

        self.GNN_layers = GNN_layers  # GCN层数
        self.lamb = sim_mutl_method['lamb']
        self.device = args.device
        self.inner_dim = inner_dim
        self.dropout = nn.Dropout(p=dropout)
        self.X_0_dim = args.input_features
        kv_dim = args.heads * inner_dim

        self.Integration_method=args.Integration_method
        self.Memory = nn.Parameter(torch.randn(1, M_dim))
        self.to_kv = nn.ModuleList()

        for i in range(GNN_layers+1):
            if i==0:
                self.to_kv.append((nn.Linear(self.X_0_dim,  kv_dim*2, bias=False)))
            else:
                self.to_kv.append(nn.Linear(X_dim, kv_dim*2, bias=False))
        self.X_out_dim=X_dim
        self.M_dim=M_dim

        self.cross_layers = M_cross_layers(num_layers=cross_MLP_layers,M_dim=M_dim,
                                           heads=args.heads, dim_heads=inner_dim,
                                           dropout=dropout)

        self.bn_feat = BatchNorm1d(self.X_0_dim)
        self.bn_M = BatchNorm1d(M_dim)
        self.bns_conv = nn.ModuleList()

        self.GNN_conv=nn.ModuleList()
        for layer in range(GNN_layers):
            X_in_dim=self.X_0_dim if layer==0 else X_dim
            GNN_model = GCNConv(Integration_method=args.Integration_method,
                                 sim_mutl_method=sim_mutl_method,lamb=self.lamb,in_channels=X_in_dim,out_channels=X_dim, M_dim=M_dim, inner_dim=self.inner_dim
                                  )
            self.GNN_conv.append(GNN_model)
            self.bns_conv.append(BatchNorm1d(X_dim))

        self.fc1 = nn.Linear(M_dim, M_dim * 2)
        self.fc2 = nn.Linear(M_dim*2, args.num_classes)
        self.ReLU = torch.nn.ReLU()



    def forward(self, dataset):

        X=dataset.x
        X=self.bn_feat(X)
        #edge_list = dataset.edge_index
        M=repeat(self.Memory, 'b d -> (g b) d', g=dataset.num_graphs)
        edge_list, _ = add_self_loops(dataset.edge_index, num_nodes=X.size(0))  # 添加自环

        #edge_list=dataset.edge_index
        for i in range(dataset.num_graphs):
            mask=torch.cat((mask,(dataset.batch==i).view(1,-1)),0) if i!=0 else (dataset.batch==i).view(1,-1)

        for layer in range(self.GNN_layers):
            M = self.cross_layers(M=M, X=X,mask=mask,to_kv=self.to_kv[layer])
            X_in_dim = self.X_0_dim if layer == 0 else self.X_out_dim
            sim_M=calucate_sim(X_in_dim,self.M_dim,self.inner_dim,X,M,mask,edge_list)

            if self.Integration_method == 'Addition':
                sim_M=self.lamb * sim_M
            elif self.Integration_method == 'Multiplication':
                sim_M= torch.pow(1 + sim_M, self.lamb)
            X_= self.GNN_conv[layer](X, edge_list,sim_M)
            X_= self.ReLU(X_)
            X=self.bns_conv[layer](X_)


        M_fin = self.cross_layers(M, X,mask=mask,to_kv=self.to_kv[-1])

        # 送入FC 分类
        M_fin2= F.relu(self.fc1(M_fin))
        M_fin2=self.dropout(M_fin2)
        classification = self.fc2(M_fin2)

        self.final_M = M_fin.detach()
        self.output = classification.detach()

        return  classification


def calucate_sim(X_in_dim,M_dim,inner_dim,X,M,mask,edge_list):
    row, col = edge_list
    to_q = nn.Linear(M_dim, inner_dim, bias=False).to(M)
    to_k = nn.Linear(X_in_dim, inner_dim, bias=False).to(M)

    q = to_q(M)
    k = to_k(X)

    sim = einsum('i d,j d -> i j', q, k) * (inner_dim ** -0.5)

    mask_sim = sim.masked_fill(mask == False, -1e9)
    mask_sim = F.softmax(mask_sim, dim=-1)
    fuzhi = torch.sum(mask_sim, dim=0)
    mask_sim= fuzhi[row]
    return mask_sim











