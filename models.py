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
class WeightNormalizedLinear(nn.Module):

    def __init__(self, in_features, out_features, scale=False, bias=False, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)

        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
class BlockGen(nn.Module):
    def __init__(self,embed_dim,num_heads,hidden):
        super().__init__()
        self.fc0 = nn.Linear(embed_dim, hidden)
        self.fc1 = nn.Linear(hidden, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc0_cls = nn.Linear(embed_dim, hidden)
        self.fc1_cls = nn.Linear(2*hidden, hidden)

        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.act = nn.GELU()
        self.ln=nn.LayerNorm(hidden)
        self.lnx2=nn.LayerNorm(embed_dim)
        self.lnx=nn.LayerNorm(hidden)
        self.context_layer = nn.Linear(embed_dim, hidden)

    def forward(self,x,x_cls=None,z=None,mask=None):
        # if mask is None:
        #     mask = torch.zeros(x.shape[0],x.shape[1],device=x.device).bool()
        res = x.clone()
        x=self.fc0(x)
        x_cls=self.fc0_cls(x_cls)
        #x = self.lnx(x)
        x[mask]*=0
        x_cls = self.ln(x_cls)
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        # x_cls = self.act(self.fc1_cls(torch.cat((x_cls,x.sum(1).unsqueeze(1)/70),dim=-1)))
        #F.glu(
        #(x_cls-x_cls.mean(0))/(x_cls.std(dim=0))
        # x = self.pre_attn_norm(x)
        x=x+x_cls.expand(-1, x.shape[1], -1).clone()#self.lnx()
        #x =F.glu(torch.cat((x, self.context_layer(x_cls.expand(-1, x.shape[1], -1).clone())), dim=-1), dim=-1)
        x_cls = self.act(self.fc2_cls(x_cls) )
        x = self.lnx(x)
        x = self.act(self.fc1(x))
        # x =self.lnx2(x+res)
        # x = self.act(self.fc2(x))
        # x=F.glu(torch.cat((x, self.context_layer(x_cls).expand(-1, x.shape[1], -1).clone()), dim=-1), dim=-1)
        # x = self.act(res+self.fc1(x))
        return x,x_cls


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen,cond_dim, **kwargs):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim_gen)#+cond_dim
        self.embbed_cond = nn.Linear(cond_dim, l_dim_gen)
        self.comb = nn.Linear(2*l_dim_gen, l_dim_gen)
        self.encoder = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=hidden_gen) for i in range(num_layers_gen)])
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.cls_token = nn.Parameter(torch.ones(1, 1, l_dim_gen), requires_grad=True)
        self.act = nn.GELU()

        self.apply(self._init_weights)

    def forward(self, x, x_cls, mask=None,mean_field=None):
        #x = self.act(self.embbed(torch.cat((x,x_cls.expand(-1,x.shape[1],-1)),dim=-1)))


        x_cls=self.act(self.embbed_cond(x_cls))
        x = self.act(self.embbed(x))
        x[mask]*=0
        # x_cls = self.act(self.comb(torch.cat((x_cls,x.sum(1).unsqueeze(1)/70),dim=-1)))
        # x_cls = self.cls_token.expand(x.size(0), 1, -1).clone()
        for layer in self.encoder:
            res=x_cls.clone()
            x, x_cls = layer(x,x_cls=x_cls,mask=mask,)
            x_cls=x_cls+res
        if mean_field is not None:
            return self.out(x), x_cls
        else:
            return self.out(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,)
            nn.init.zeros_(m.bias,)
        if isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal_(m.in_proj_weight)


class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden):
        super().__init__()
        self.fc0 = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1 = (WeightNormalizedLinear( hidden,embed_dim))
        self.fc0_cls = (WeightNormalizedLinear(embed_dim, hidden))
        self.fc1_cls = (WeightNormalizedLinear(2*hidden,hidden))
        self.fc2_cls = WeightNormalizedLinear(hidden, embed_dim)
        self.attn = weight_norm(nn.MultiheadAttention(hidden, num_heads, batch_first=True, dropout=0),"in_proj_weight")
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln_cls = nn.LayerNorm(embed_dim)
        self.context_layer = nn.Linear(hidden, hidden)
        self.hidden = hidden

    def forward(self, x, x_cls, src_key_padding_mask=None):
        #res=x.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.fc0_cls(x_cls))
        x = self.ln(x)
        # x_cls = self.ln(x_cls)
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x=x+x_cls.expand(-1, x.shape[1], -1).clone()
        # x=F.glu(torch.cat((x, self.context_layer(x_cls).expand(-1, x.shape[1], -1).clone()), dim=-1), dim=-1)


        #x= self.bn(x.reshape(-1,x.shape[-1])).reshape(x.shape)
        x[src_key_padding_mask]*=0
        # x_cls = self.act(self.fc1_cls(torch.cat((x_cls,x.sum(1).unsqueeze(1)/70),dim=-1)))#+x.mean(dim=1).unsqueeze(1)
        x_cls = self.act(self.fc2_cls(x_cls))
        # x_cls=self.ln_cls(x_cls)
        x=self.fc1(x)
        # x=self.ln2(x+res)
        return x_cls,x


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList([BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden) for i in range(num_layers)])
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(2*l_dim, l_dim)
        self.fc2 = WeightNormalizedLinear(l_dim, l_dim)
        self.ln=nn.LayerNorm(l_dim)
        self.apply(self._init_weights)

    def forward(self, x, mask=None,mean_field=False):
        x = self.act(self.embbed(x))
        x[mask]*=0
        x_cls = x.sum(1).unsqueeze(1).clone()/70


        for layer in self.encoder:

            res=x_cls.clone()
            x_cls,x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
            x_cls=(x_cls+res)


        x_cls = self.act(self.fc1(torch.cat((x_cls.squeeze(1),x.sum(1)/70),dim=-1)))

        x_cls=self.ln(x_cls)
        res=x_cls.clone()
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
    z = torch.randn(1000, 150, 3)
    z_cls = torch.randn(1000, 1, 6)
    model = Gen(3, 6, 128, 3, 8,6)
    masks=torch.zeros(1000,150).bool()
    a=model(x=z,x_cls=z_cls,mask=masks)
    print(a.mean(),a.std())
    model = Disc(3, 6, 128, 3, 8)
    b=model(z,mask=masks)
    print(b.mean(),b.std())