import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

class BlockGen(nn.Module):
    def __init__(self,embed_dim,num_heads,):
        super().__init__()
        self.fc0 = weight_norm(nn.Linear(embed_dim, embed_dim))
        self.fc1 = weight_norm(nn.Linear(embed_dim, embed_dim))
        self.fc2 = weight_norm(nn.Linear(embed_dim, embed_dim))
        self.fc1_cls = weight_norm(nn.Linear(embed_dim, embed_dim))
        self.fc2_cls = weight_norm(nn.Linear(embed_dim, embed_dim))
        self.attn = weight_norm(nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),"in_proj_weight")
        self.act = nn.LeakyReLU()

    def forward(self,x,x_cls=None,z=None,mask=None):
        # if mask is None:
        #     mask = torch.zeros(x.shape[0],x.shape[1],device=x.device).bool()
        res = x.clone()
        x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
        x_cls = self.act(self.fc1_cls((x_cls-x_cls.mean(0))/(x_cls.std(dim=0))))
        x_cls = self.act(self.fc2_cls(x_cls) )
        # x = self.pre_attn_norm(x)
        x =x_cls.expand(-1, x.shape[1], -1).clone()+res
        x = self.act(self.fc1(x))
        #x = self.act(self.fc2(x))
        return x,x_cls


class Gen(nn.Module):
    def __init__(self, n_dim, l_dim_gen, hidden_gen, num_layers_gen, heads_gen, **kwargs):
        super().__init__()
        l_dim_gen = hidden_gen
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.encoder = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen) for i in range(num_layers_gen)])
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.cls_token = nn.Parameter(torch.ones(1, 1, l_dim_gen), requires_grad=True)
        self.act = nn.LeakyReLU()

        self.apply(self._init_weights)

    def forward(self, x, mask=None,mean_field=None):
        x = self.act(self.embbed(x))
        x_cls = self.cls_token.expand(x.size(0), 1, -1).clone()
        for layer in self.encoder:
            x, x_cls = layer(x,x_cls=x_cls,mask=mask,)
        if mean_field is not None:
            return self.out(1.2*x), x_cls
        else:
            return self.out(1.2*x)

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
        self.fc0 = (nn.Linear(embed_dim, hidden))
        self.fc1_cls = (nn.Linear(embed_dim, hidden))
        self.fc2_cls = nn.Linear(hidden, embed_dim)
        self.attn = weight_norm(nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0), "in_proj_weight")
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.hidden = hidden

    def forward(self, x, x_cls, src_key_padding_mask=None):
        res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x=x+x_cls
        #x= self.bn(x.reshape(-1,x.shape[-1])).reshape(x.shape)
        x_cls = self.act(self.fc1_cls(x_cls))#+x.mean(dim=1).unsqueeze(1)
        x_cls = self.act(self.fc2_cls(x_cls+res))
        return x_cls,x


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, **kwargs):
        super().__init__()
        l_dim = hidden
        self.embbed = nn.Linear(n_dim, l_dim)
        self.encoder = nn.ModuleList([BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden) for i in range(num_layers)])
        self.out = weight_norm(nn.Linear(l_dim, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = weight_norm(nn.Linear(l_dim, hidden))
        self.fc2 = weight_norm(nn.Linear(hidden, l_dim))
        self.apply(self._init_weights)

    def forward(self, x, mask=None,mean_field=False):
        x = self.act(self.embbed(x))
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        for layer in self.encoder:
            x_cls,x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
        res=x_cls.clone()
        x_cls = self.act(self.fc1(x_cls))
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
    model = Gen(3, 128, 128, 3, 8)
    print(model(z).std())
    model = Disc(3, 128, 128, 3, 8)
    print(model(z).std())
