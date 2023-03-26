import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import torch.nn.functional as F
from helpers import WeightNormalizedLinear
class BlockGen(nn.Module):
    def __init__(self,embed_dim,hidden,num_heads,act):
        super().__init__()
        self.fc0 = WeightNormalizedLinear(embed_dim, hidden)
        self.fc1 = WeightNormalizedLinear(2*hidden, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, embed_dim)
        self.fc1_cls = WeightNormalizedLinear(2*hidden, hidden)
        # self.fc2_cls = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)#,"in_proj_weight")
        self.context=nn.Linear(embed_dim,embed_dim)
        self.act = nn.LeakyReLU() if act=="leaky" else nn.GELU()
        self.ln=nn.LayerNorm(hidden)
        self.glu=nn.GLU(dim=-1)
    def forward(self,x,x_cls=None,z=None,mask=None):
        # if mask is None:
        #     mask = torch.zeros(x.shape[0],x.shape[1],device=x.device).bool()
        res = x.clone()
        x = self.act(self.fc0(x))
        x=x*((~mask).unsqueeze(-1).float())
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        x_cls = self.act(self.ln(self.fc1_cls((torch.cat((x.sum(1).unsqueeze(1),x_cls),dim=-1)))))
        x = self.act(self.fc1((torch.cat((x,x_cls.expand_as(x)),dim=-1))))
        x = self.act(self.fc2(x)+res)
        return x,x_cls


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, cond_dim, act, **kwargs):
        super().__init__()
        l_dim_gen = hidden_gen
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.embbed_cond = nn.Linear(cond_dim, l_dim_gen)
        self.encoder = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen,act=act) for i in range(num_layers_gen)])
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_gen), requires_grad=True)
        self.act = nn.LeakyReLU() if act=="leaky" else nn.GELU()
        self.apply(self._init_weights)

    def forward(self, x, x_cls, mask=None,mean_field=None):
        x = self.act(self.embbed(x))
        mask=mask.bool()
        if x_cls is None:
            x_cls = self.cls_token.expand(x.size(0), 1, -1)#.clone()
        else:
            x_cls = self.act(self.embbed_cond(x_cls))
        for layer in self.encoder:
            x, x_cls = layer(x,x_cls=x_cls,mask=mask,)
        return self.out(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,)
        if isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal_(m.in_proj_weight)
        torch.nn.init.kaiming_normal_(self.embbed.weight,)
        torch.nn.init.kaiming_normal_(self.out.weight,)

class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden):
        super().__init__()
        self.fc0 = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1_cls = (WeightNormalizedLinear(2*hidden, hidden))
        self.fc2_cls = WeightNormalizedLinear(hidden, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0)
        self.act = nn.LeakyReLU()
        self.ln=nn.LayerNorm(hidden)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x, x_cls, mask=None):
        #res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.fc0(x_cls))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        x[mask<=-10]=0
        x_cls = self.act(self.ln(self.fc1_cls(torch.cat((x_cls,x.sum(1).unsqueeze(1)),-1))))
        #x_cls = self.act(self.ln(self.fc1_cls(torch.cat((x_cls,self.act(self.sum(x.sum(1))).unsqueeze(1)),-1))))#+x.mean(dim=1).unsqueeze(1)
        x_cls = self.fc2_cls(x_cls)
        return x_cls,x


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, **kwargs):
        super().__init__()
        l_dim = hidden
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        #self.embbed_cls = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList([BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.apply(self._init_weights)

    def forward(self, x, mask=None,mean_field=False):
        x = self.act(self.embbed(x))
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        for layer in self.encoder:
            x_cls=self.act(x_cls)
            x_cls,x = layer(x, x_cls=x_cls, mask=mask)

        res=x_cls.clone()
        x_cls=self.act(x_cls)
        x_cls=x_cls#+x.sum(1).unsqueeze(1)
        x_cls=self.act(x_cls)
        x_cls = self.act(self.fc1(x_cls))
        x_cls = self.act(self.fc2(x_cls))
        if mean_field:
            return self.out(x_cls),res
        else:
            return self.out(x_cls)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal_(m.in_proj_weight)
        torch.nn.init.kaiming_normal_(self.embbed.weight,)
        torch.nn.init.kaiming_normal_(self.out.weight,)





if __name__ == "__main__":
    z = torch.randn(10000, 150, 3)
    mask = torch.randint(0, 1, (10000, 150))
    x_cls = torch.randn(10000, 1, 6)
    model = Gen(3, 128, 128, 6, 8,6)
    print(model(z,x_cls, mask).std())
    model = Disc(3, 128, 128, 3, 8)
    print(model(z,mask).std())