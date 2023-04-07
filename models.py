import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter

from helpers import TPReLU, WeightNormalizedLinear


class BlockGen(nn.Module):
    def __init__(self,embed_dim,hidden,num_heads,):
        super().__init__()
        self.fc0 = nn.Linear(embed_dim, hidden)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(hidden+embed_dim, embed_dim)
        self.fc1_cls = nn.Linear(hidden+1, embed_dim)
        #self.fc2_cls = WeightNormalizedLinear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self,x,x_cls,mask):
        # if mask is None:
        #     mask = torch.zeros(x.shape[0],x.shape[1],device=x.device).bool()
        res = x.clone()
        x=self.act(self.fc0(x))
        x_cls = self.fc0(x_cls)
        x_cls=self.act(self.ln(x_cls))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        x_cls = self.fc1_cls(torch.cat((x_cls,mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1))
        #x_cls = self.act(self.fc2_cls(x_cls) )
        # x = self.pre_attn_norm(x)
        x=self.act(self.fc2(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1)))
        x = self.act(self.fc1(x)+res)
        #x = self.act(self.fc2(x))
        return x,x_cls


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, **kwargs):
        super().__init__()
        #l_dim_gen = hidden_gen
        self.embbed = nn.Linear(n_dim, l_dim_gen)

        self.encoder = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen) for i in range(num_layers_gen)])
        self.out = WeightNormalizedLinear(l_dim_gen, n_dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_gen), requires_grad=True)
        self.act = nn.LeakyReLU()

        # self.apply(self._init_weights)

    def forward(self, x,mask):
        x = self.act(self.embbed(x))
        # x[mask]*=0
        x_cls = x.sum(1).unsqueeze(1).clone()/70#
       # x_cls = self.cls_token.expand(x.size(0), 1, -1).clone()
        for layer in self.encoder:
            x, x_cls = layer(x,x_cls=x_cls,mask=mask,)
        # if mean_field is not None:
        return self.out(x)#, x_cls
        # else:
        #     return self.out(x)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight,)
    #     if isinstance(m, nn.MultiheadAttention):
    #         nn.init.kaiming_normal_(m.in_proj_weight)
    #     torch.nn.init.kaiming_normal_(self.embbed.weight,)
    #     torch.nn.init.kaiming_normal_(self.out.weight,)
class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden,dropout):
        super().__init__()
        self.fc0 = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1 = (WeightNormalizedLinear(hidden+embed_dim, embed_dim))

        self.fc0_cls = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1_cls = (WeightNormalizedLinear(hidden+1, embed_dim))
        # self.fc2_cls = WeightNormalizedLinear(hidden, embed_dim)
        #self.fc2 = WeightNormalizedLinear(2*hidden, embed_dim)
        self.attn = weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=dropout),"in_proj_weight")
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, x, x_cls, mask):
        res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.ln(self.fc0_cls(x_cls)))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        x_cls = self.act(self.fc1_cls(torch.cat((x_cls,mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1)))#+x.mean(dim=1).
        x=self.act(self.fc1(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1)))
        x_cls =x_cls+res
        return x_cls,x


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads,dropout, **kwargs):
        super().__init__()
        # l_dim = hidden
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList([BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden,dropout=dropout) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim+1, l_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.fc1_m = WeightNormalizedLinear(l_dim, hidden)
        self.fc2_m = WeightNormalizedLinear(hidden, 1)
        self.ln = nn.LayerNorm(l_dim)
        # self.apply(self._init_weights)

    def forward(self, x, mask,):#mean_field=False
        x = self.act(self.embbed(x))
        #x[mask]*=0
        x_cls = torch.cat((x.mean(1).unsqueeze(1).clone(),mask.sum(1).unsqueeze(1).unsqueeze(1)/70),dim=-1)# self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x_cls,x = layer(x, x_cls=x_cls, mask=mask)
            res=x_cls.clone()
            x_cls=(self.act(x_cls))
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls)))))
        m=self.fc2_m(self.act(self.fc1_m(x_cls.clone().squeeze(1))))
        # if mean_field:
        return self.out(x_cls),res,m
        # else:
        #     return self.out(x_cls)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight,)
    #         m.bias.data.fill_(0.01)
    #     if isinstance(m, nn.MultiheadAttention):
    #         nn.init.kaiming_normal_(m.in_proj_weight)
    #     torch.nn.init.kaiming_normal_(self.embbed.weight,)
    #     torch.nn.init.kaiming_normal_(self.out.weight,)


if __name__ == "__main__":
    z = torch.randn(1000, 150, 3)
    mask = torch.zeros((1000, 150)).bool()
    x_cls = torch.randn(1000, 1, 6)
    model =Gen(3, 64, 128, 8,8).cuda()
    print(model(z.cuda(), mask.cuda()).std())
    model = Disc(3, 64, 128, 3, 8,0.1).cuda()

    #print(model(z.cuda(),mask.cuda()))