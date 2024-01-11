
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
import math, copy, time
from torch.nn import BatchNorm1d

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    实现子层连接结构的类 res+norm
    """
    def __init__(self, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, M,x,mask,to_kv,layer):

        out_nodes = layer(M, x, mask, to_kv)

        out_nodes = self.dropout(out_nodes)
        if out_nodes.shape[-1] == M.shape[-1]:
            out_nodes += M
        else:
            raise ValueError('args.dim_heads改成和M_dim一致')

        out_nodes = F.layer_norm(out_nodes, [out_nodes.size()[-1]])


        return out_nodes


class cross_layer(nn.Module):
    def __init__(self,M_dim,heads, dim_heads, dropout):
        super(cross_layer, self).__init__()
        self.self_attn = cross_att(M_dim,heads, dim_heads)
        self.feed_forward =FeedForward( M_dim, mult=4,dropout=0.1)
        self.sublayer = clones(SublayerConnection(dropout), 2)


    def forward(self,M, x,mask,to_kv):
        # attention sub layer
        m = self.sublayer[0](M,x,mask,to_kv,self.self_attn)
        # feed forward sub layer
        z = self.sublayer[1](m,None,None, None,self.feed_forward)
        return z

class cross_att(nn.Module):
    def __init__(self, M_dim, heads, dim_heads):
        super(cross_att, self).__init__()
        self.M_dim = M_dim
        self.heads = heads
        self.dim_heads = dim_heads

        self.inner_dim = dim_heads * heads

        self.to_q =nn.Linear(M_dim, self.inner_dim, bias=False)

        self.mutl_heads =nn.Linear(self.inner_dim, M_dim)


    def forward(self, M, X,mask,to_kv):
        '''
        :param M: e x c
        :param X: node_num x d
        :return:
        '''

        M, X = map(lambda t: repeat(t, 'n d ->b n d', b=1), (M, X))  # 添加维度 1 x e x c / 1 x node_num x d

        q = self.to_q(M)  # 1 x e x inner
        k, v = to_kv(X).chunk(2, dim=-1)  # 1 x node_num x inner

        q, k, v = map(lambda t: rearrange(t, 'b n (h dim) -> (b h) n dim', h=self.heads),
                      (q, k, v))  # h x n x dim_heads

        sim = einsum('b i d,b j d -> b i j', q, k) * (self.dim_heads ** -0.5)

        mask_sim=sim.masked_fill(mask==False,-1e9)
        mask_sim=F.softmax(mask_sim, dim=-1)

        out_M = einsum('b i j, b j d -> b i d', mask_sim, v)
        out_M = rearrange(out_M, '(b h) n d -> b n (h d)', h=self.heads)
        out_M=out_M.squeeze(0)

        out=self.mutl_heads(out_M)

        return out




class FeedForward(nn.Module):
    def __init__(self, M_dim, mult=4,dropout=0.1):
        #初始化函数有三个输入参数分别是d_model，d_ff，和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，因为我们希望输入通过前馈全连接层后输入和输出的维度不变，第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出，最后一个是dropout置0比率。
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(M_dim, M_dim * mult)
        self.w_2 = nn.Linear(M_dim * mult, M_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = BatchNorm1d(M_dim * mult)

    def forward(self, out,N1,N2,N3):
        #输入参数为x，代表来自上一层的输出，首先经过第一个线性层，然后使用F中的relu函数进行激活，之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果

        return self.w_2(self.dropout(F.relu(self.w_1(out))))



class M_cross_layers(nn.Module):
    def __init__(self,num_layers,M_dim,heads=8, dim_heads=64,dropout=0.0):
        super(M_cross_layers, self).__init__()
        self.num_layers = num_layers
        self.M_dim = M_dim

        self.heads = heads
        self.dim_heads = dim_heads

        self.layer_stack = nn.ModuleList([
            cross_layer(M_dim,heads, dim_heads, dropout)
            for _ in range(num_layers)])

    def forward(self, M, X, mask,to_kv):
        '''块内不参数共享'''

        for layer in self.layer_stack:
            output =layer(M,X,mask,to_kv)
            M=output

        return M







