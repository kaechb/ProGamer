import torch
from helpers import to_canonical
from particle_attention import *
from torch import nn
from torch.nn.functional import leaky_relu, sigmoid
from helpers import CosineWarmupScheduler, Scheduler, EqualLR,equal_lr
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from monotonenorm import direct_norm


def leaky(x,slope):
    return leaky_relu(x, slope)

def masked_layer_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads,hidden,
                 dropout, activation,slope,norm,proj,spectral):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = hidden
        
        self.slope=slope
        self.norm=norm
        self.proj=proj
        
        
        if spectral:
            self.attn = spectral_norm(nn.MultiheadAttention(embed_dim,num_heads,batch_first=True,dropout=0),"in_proj_weight")
        else:
            self.attn = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True,dropout=0)
        self.dropout = nn.Dropout(dropout)
        self.bn =nn.GroupNorm(1,embed_dim)
        self.ln =nn.LayerNorm(embed_dim)
        self.pre_fc_norm=nn.LayerNorm(embed_dim)
        self.gn=nn.GroupNorm(1,embed_dim)
        self.proj=proj
        if spectral:
            self.fc1 = spectral_norm(nn.Linear(embed_dim, self.ffn_dim))
            
            self.fc2 = spectral_norm(nn.Linear(self.ffn_dim, embed_dim))
        else:
            self.fc1 = (nn.Linear(embed_dim, self.ffn_dim))
            
            self.fc2 = (nn.Linear(self.ffn_dim, embed_dim))
        
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.hidden=hidden
        self.projection=nn.Linear(2*embed_dim,embed_dim)

    def forward(self, x, x_cls, src_key_padding_mask=None, ):
      
        
        residual = x_cls.clone()
        u = self.ln(x)
        x = self.attn(x_cls, u, u, key_padding_mask=src_key_padding_mask)[0]  
        x=self.dropout(x)
        if self.proj:
            x=self.projection(torch.cat((x,residual),axis=-1))  
        else:
            x += residual
        residual=x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x+=residual
        return x
class BlockGen(nn.Module):
    def __init__(self, embed_dim, num_heads,hidden,
                 dropout, activation,norm):
        super().__init__()
        self.hidden=hidden
        self.norm=norm
        self.embed_dim=embed_dim
        self.ffn_dim = hidden

        #self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout=0,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)
        self.projection=nn.Linear(2*embed_dim,embed_dim)
        self.bn = nn.GroupNorm(1,30)
        self.gn = nn.GroupNorm(1,hidden)
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.projection=nn.Linear(2*embed_dim,embed_dim)


    def forward(self, x, x_cls=None,z=None, src_key_padding_mask=None,):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch,seq_len, embed_dim)`
        """
        if z is  None:
            res=x
        else:
            res=z
        x = self.pre_attn_norm(x)
        x = x_cls.expand(-1,x.shape[1],-1)
        x = self.dropout(x)
        x = x+res
        res = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += res

        return x

class Gen(nn.Module):
    def __init__(self,n_dim,l_dim_gen,hidden_gen,num_layers_gen,heads_gen,dropout_gen,activation_gen,n_part,slope,res,norm,random,proj_gen,spectral,**kwargs):
        super().__init__()
        self.slope=slope
        self.proj_gen=proj_gen
        self.embbed = nn.Linear(n_dim, l_dim_gen)
        self.random=random
        self.hidden_nodes = int(hidden_gen)
        self.n_dim = n_dim
        self.n_part=n_part
        self.res=res
       
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=self.hidden_nodes,dropout=dropout_gen,activation=activation_gen,norm=norm,proj=proj_gen,spectral=spectral,slope=slope)
            for i in range(num_layers_gen)])
        self.encoder_gen = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=self.hidden_nodes,dropout=dropout_gen,activation=activation_gen,norm=norm) for i in range(num_layers_gen)])
        
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(l_dim_gen, n_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)
        self.reduce_embbed = nn.Linear(2*l_dim_gen, l_dim_gen)
        self.act = nn.GELU()
        
        

    def forward(self, x, mask=None,):
        
        x = self.embbed(x)
        if self.random:
            x_cls=torch.randn_like(x[:,:1,:])
        else:
            x_cls=self.cls_token.expand( x.size(0),1, -1)
        z = x.clone()
        for cls_layer,layer in zip(self.encoder,self.encoder_gen):
            x_cls_post = cls_layer(x,x_cls,src_key_padding_mask=mask)#attention_mask.bool()
            if not self.res:
                z=None
            x_cls=self.reduce_embbed(torch.cat((x_cls,x_cls_post),axis=-1))
            x = layer(x,x_cls=x_cls,z=z,src_key_padding_mask=mask.bool() )
        
        return self.out(x)
      
class Disc(nn.Module):
    def __init__(self,n_dim,l_dim,hidden,num_layers,heads,dropout,activation,slope,norm,spectral,proj,**kwargs):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim)
        self.reduce_embbed=nn.Linear(2*l_dim,l_dim)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=heads,hidden=int(hidden),dropout=dropout,activation=activation,slope=0.1,norm=norm,proj=proj,spectral=spectral) for i in range(num_layers)])
        if spectral:
            self.hidden = spectral_norm(nn.Linear(l_dim ,hidden ))
            self.hidden2 = spectral_norm(nn.Linear(int(hidden), l_dim ))
        else:
            self.hidden = (nn.Linear(l_dim ,2*hidden ))
            self.hidden2 = (nn.Linear(2*hidden, l_dim ))
        self.out = nn.Linear(l_dim , 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)
        

    def forward(self, x, mask=None):
        
        x = self.embbed(x)
        x_cls=self.cls_token.expand( x.size(0),1, -1)
        for layer in self.encoder:
            x_cls_post = layer(x,x_cls,src_key_padding_mask=mask )#attention_mask.bool()
            x_cls=self.reduce_embbed(torch.cat((x_cls,x_cls_post),axis=-1))
        x=x_cls.reshape(len(x),x.shape[-1])

        x = leaky(self.hidden(x), 0.2)
        x = leaky(self.hidden2(x), 0.2)
        return self.out(x)



