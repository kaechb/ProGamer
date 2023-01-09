import torch
from helpers import to_canonical
from particle_attention import *
from torch import nn
from torch.nn.functional import leaky_relu, sigmoid
from helpers import CosineWarmupScheduler, Scheduler, EqualLR,equal_lr

def leaky(x,slope):
    return leaky_relu(x, slope)
class Block(nn.Module):
    def __init__(self, embed_dim=60, num_heads=4,hidden=60,
                 dropout=0.1, activation='relu',proj=True,affine=False,slope=0.1,bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = hidden
        self.proj=proj
        
        self.reduce_embbed=nn.Linear(2*embed_dim,embed_dim)
        self.slope=slope
        self.pre_attn_norm = nn.LayerNorm(embed_dim,elementwise_affine=affine)
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True,bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(embed_dim,elementwise_affine=affine)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.hidden=hidden
    def forward(self, x, x_cls=None, src_key_padding_mask=None, attn_mask=None):
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

        residual = x_cls
        u=x
        u = self.pre_attn_norm(x)
        
        x = self.attn(x_cls, u, u, key_padding_mask=src_key_padding_mask)[0]  # ( batch,1, embed_dim)
        x += residual
        residual = x
        if self.hidden:
            x = self.pre_fc_norm(x)
            x = leaky(self.fc1(x),self.slope)
            x = self.act_dropout(x)
            x = self.fc2(x)+residual
        x=leaky(self.reduce_embbed(torch.cat((x_cls,x),axis=-1)),self.slope)
        return x
class BlockGen(nn.Module):
    def __init__(self, embed_dim, num_heads,hidden,
                 dropout, activation,proj,affine,bias):
        super().__init__()
        self.hidden=hidden
        self.proj=proj
        embed_dim=embed_dim
        self.ffn_dim = hidden
        self.pre_attn_norm = nn.LayerNorm(embed_dim,elementwise_affine=affine)
        #self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout=0,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(embed_dim,elementwise_affine=affine)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim,bias=bias)
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim,bias=bias)
        self.projection=nn.Linear(2*embed_dim,embed_dim,bias=bias)
        

    def forward(self, x, x_cls=None, src_key_padding_mask=None, attn_mask=None):
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
        
        residual = x
        
        x = self.pre_attn_norm(x)
        #x = self.attn( x,x_cls, x_cls, attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
        x = x_cls.expand(-1,x.shape[1],-1).clone()
        #x = self.dropout(x)
        # if self.proj:
        #     x=self.projection(torch.cat((x,residual),axis=-1))
        # else:
        x += residual
        residual = x
        if self.hidden:
            x = self.pre_fc_norm(x)
            x = self.act(self.fc1(x))
            x = self.act_dropout(x)
            x = self.fc2(x)+residual
        

        return x

class Gen(nn.Module):
    def __init__(self,n_dim,l_dim_gen,hidden_gen,num_layers_gen,heads_gen,proj_gen,dropout_gen,activation_gen,latent,n_part,affine,bias,slope,**kwargs):
        super().__init__()
        self.slope=slope
        
        self.embbed = nn.Linear(n_dim, l_dim_gen,bias=bias)
        self.latent=latent
        self.hidden_nodes = int(hidden_gen)
        self.n_dim = n_dim
        self.n_part=n_part
        if latent:
            self.up=nn.Linear(latent,n_dim*n_part)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=self.hidden_nodes,dropout=dropout_gen,proj=proj_gen,activation=activation_gen,affine=affine,bias=bias)
            for i in range(num_layers_gen)])
        self.encoder_gen = nn.ModuleList([BlockGen(embed_dim=l_dim_gen, num_heads=heads_gen,hidden=self.hidden_nodes,dropout=dropout_gen,proj=proj_gen,activation=activation_gen,affine=affine,bias=bias) for i in range(num_layers_gen)])
        
        self.dropout = nn.Dropout(dropout_gen)
        self.out = nn.Linear(l_dim_gen, n_dim,bias=bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim_gen), requires_grad=True)

        self.act = nn.GELU()
        
        

    def forward(self, x, mask=None):
        
        x_cls=self.cls_token.expand( x.size(0),1, -1).clone()
        x = self.embbed(x)
        for cls_layer,layer in zip(self.encoder,self.encoder_gen):
            x_cls = cls_layer(x,x_cls,src_key_padding_mask=mask.bool())#attention_mask.bool()
            x = layer(x,x_cls,src_key_padding_mask=mask.bool() )
        return self.out(x)
      
class Disc(nn.Module):
    def __init__(self,n_dim,l_dim,hidden,num_layers,heads,dropout,proj,activation,slope,affine,bias,**kwargs):
        super().__init__()
        
        self.embbed = nn.Linear(n_dim, l_dim,bias=bias)
        
        self.slope=slope

        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=heads,hidden=int(hidden),dropout=dropout,proj=proj,activation=activation,slope=0.1,bias=bias,affine=affine)
            for i in range(num_layers)])
        self.hidden = nn.Linear(l_dim , int(hidden),bias=bias )

        self.hidden2 = nn.Linear(int(hidden), l_dim,bias=bias )

        self.fc_aux= nn.Linear(l_dim , l_dim )
        self.batchnorm1=nn.BatchNorm1d(hidden)
        self.batchnorm2=nn.BatchNorm1d(l_dim)

        self.fc2_aux = nn.Linear(l_dim , l_dim )
        self.out = nn.Linear(l_dim , 1,bias=bias)
        self.out_aux = nn.Linear(l_dim , 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)


    def forward(self, x, mask=None,aux=False):
        
        x = self.embbed(x)
        x_cls=self.cls_token.expand( x.size(0),1, -1).clone()
        
        for layer in self.encoder:

            x_cls = layer(x,x_cls,src_key_padding_mask=mask.bool() )#attention_mask.bool()
            
        x=x_cls.reshape(len(x),x.shape[-1])
        if aux:
            temp=leaky(self.fc_aux(x),self.slope)
            temp=leaky(self.fc2_aux(temp),self.slope)
            temp=self.out_aux(temp)
            m=temp[:,0]
            p=temp[:,1]
        res=x
        x = leaky(self.hidden(x), self.slope)
        #x = leaky(self.hidden(x), self.slope)
        x = leaky(self.hidden2(x), self.slope)+res
        # x = leaky(self.batchnorm(self.hidden(x)), self.slope)
        # x = leaky(self.batchnorm2(self.hidden2(x)), self.slope)
        if aux:
            return self.out(x),m,p
        else:
            return self.out(x)



