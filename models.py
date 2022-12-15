import torch
from helpers import to_canonical
from particle_attention import *
from torch import nn
from torch.nn.functional import leaky_relu, sigmoid


def leaky(x):
    return leaky_relu(x, 0.01)
class Block(nn.Module):
    def __init__(self, embed_dim=60, num_heads=4,hidden=512,
                 dropout=0.1, activation='relu',norm_first=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = hidden
        self.norm_first=norm_first
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)
        
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

        if x_cls is not None:
           
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                #src_key_padding_mask = torch.cat((torch.zeros_like(src_key_padding_mask[:, :1]), src_key_padding_mask), dim=1).bool()
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            # u = torch.cat((x_cls, x), dim=1)  # (seq_len+1, batch, embed_dim)
            # u = torch.cat((x_cls, x), dim=1)  # (seq_len+1, batch, embed_dim)

            u = self.pre_attn_norm(x)
            x = self.attn(x_cls, u, u, key_padding_mask=src_key_padding_mask)[0]  # ( batch,1, embed_dim)
            if not self.norm_first:
                x = self.post_attn_norm(x)
        else:
            residual = x
            if True:
                x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=src_key_padding_mask,
                          attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
        x = self.dropout(x)
        x += residual
        residual = x
        
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))

        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += residual
        return x
class BlockGen(nn.Module):
    def __init__(self, embed_dim=60, num_heads=4,hidden=512,
                 dropout=0.1, activation='relu',norm_first=True):
        super().__init__()
        self.norm_first=norm_first
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = hidden
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)
        
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
        x = self.attn( x,x_cls, x_cls, attn_mask=attn_mask)[0]  # (seq_len, batch, embed_dim)
        
        
        #x = x_cls.expand(-1,x.shape[1],-1)
        x = self.dropout(x)
        x += residual
        residual = x
        x = self.pre_fc_norm(x)
        
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += residual
        return x

class Gen(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,norm_first=True,dropout=0.5,activation="leakyrelu"):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim

        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=num_heads,hidden=hidden,dropout=dropout,norm_first=norm_first,activation=activation)
            for i in range(num_layers)])
        self.encodergen = nn.ModuleList([BlockGen(embed_dim=l_dim, num_heads=num_heads,hidden=hidden,dropout=dropout,norm_first=norm_first,activation=activation)
            for i in range(num_layers)])
        self.embbed = nn.Linear(n_dim, l_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(l_dim, n_dim)
        self.out_mom = nn.Linear(l_dim , 1)
        self.out_mass = nn.Linear(l_dim , 1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)
        self.reduce_embbed = nn.Linear(2*l_dim, l_dim)

        

    def forward(self, x, mask=None):

        x_cls=self.cls_token.expand( x.size(0),1, -1)
        x = self.embbed(x)
        for cls_layer,layer in zip(self.encoder,self.encodergen):

            x_cls_post = cls_layer(x,x_cls,src_key_padding_mask=mask.bool() )#attention_mask.bool()
            x_cls=self.reduce_embbed(torch.cat((x_cls,x_cls_post),axis=-1))
            x = layer(x,x_cls,src_key_padding_mask=mask.bool() )

      
        return self.out(x)
      


class Disc(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,dropout=0.5,norm_first=True,activation="leakyrelu"):
        super().__init__()
  
        self.embbed = nn.Linear(n_dim, l_dim)
        self.reduce_embbed = nn.Linear(2*l_dim, l_dim)
        self.encoder = nn.ModuleList([Block(embed_dim=l_dim, num_heads=num_heads,hidden=hidden,dropout=dropout,norm_first=norm_first,activation=activation)
            for i in range(num_layers)])
        self.hidden = nn.Linear(l_dim , 2 * hidden )
        self.hidden2 = nn.Linear(2 * hidden , l_dim )
        self.out = nn.Linear(l_dim , 1)
        self.out_aux = nn.Linear(l_dim , 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)
      
    def forward(self, x, mask=None,aux=False):

        x = self.embbed(x)
        x_cls=self.cls_token.expand( x.size(0),1, -1)
        for layer in self.encoder:
            x_cls_post = layer(x,x_cls,src_key_padding_mask=mask.bool() )#attention_mask.bool()
            x_cls=self.reduce_embbed(torch.cat((x_cls,x_cls_post),axis=-1))
        x=x_cls.reshape(len(x),x.shape[-1])
        if aux:
            temp=self.out_aux(x)
            m=temp[:,0]
            p=temp[:,1]
       
        x = leaky_relu(self.hidden(x), 0.01)
        x = leaky_relu(self.hidden2(x), 0.01)
        if aux:
            return self.out(x),m,p
        else:
            return self.out(x)
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        



