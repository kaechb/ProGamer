from torch import nn
import torch
from torch.nn.functional import leaky_relu, sigmoid

def leaky(x):
    return leaky_relu(x, 0.2)
class Block(nn.Module):
    def __init__(self, embed_dim=60, num_heads=4,hidden=512,
                 dropout=0.1, activation='relu',):
        super().__init__()

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

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
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
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=1)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
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

class PairEmbed(nn.Module):
    def __init__(self, input_dim, dim=64, normalize_input=True, activation='leaky', eps=1e-8, for_onnx=False):
        super().__init__()

        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=input_dim, eps=eps, for_onnx=for_onnx)

        module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.LeakyReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

        self.out_dim = dim

    def forward(self, x):
        # x: (batch, v_dim, seq_len)
        x=x.permute(0,2,1)
        with torch.no_grad():
            batch_size, _, seq_len = x.size()
            x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2)).view(batch_size, -1, seq_len * seq_len)

        elements = self.embed(x)  # (batch, embed_dim, num_elements)
        y = elements.view(batch_size, -1, seq_len, seq_len)
        y=y.permute(0,3,1,2)
        return y

class Gen(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,norm_first=False,dropout=0.5,no_hidden=True,):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        
        self.no_hidden_gen = no_hidden
        self.embbed = nn.Linear(n_dim, l_dim)
   

        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=l_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden,
                    dropout=dropout,
                    batch_first=True,
                    activation=leaky,
                    norm_first=norm_first
                ),
                num_layers=num_layers,
            )
        if not self.no_hidden_gen==True:
            self.hidden = nn.Linear(l_dim, hidden)
            self.hidden2 = nn.Linear(hidden, hidden)
        if self.no_hidden_gen=="more":
            self.hidden3 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout / 2)
        self.out = nn.Linear(hidden, n_dim)
        self.out2 = nn.Linear(l_dim, n_dim)
        
        

    def forward(self, x, mask=None):
        x = self.embbed(x)
        x = self.encoder(x, src_key_padding_mask=mask.bool())#attention_mask.bool()
        if self.no_hidden_gen=="more":
            x = leaky_relu(self.hidden(x))
            x = self.dropout(x)
            x = leaky_relu(self.hidden2(x))
            x = self.dropout(x)
            x = leaky_relu(self.hidden3(x))
            x = self.dropout(x)  
            x=self.out(x)
        elif self.no_hidden_gen==True:
            #x = leaky_relu(x)
            x = self.out2(x)
        else:
            x = leaky_relu(self.hidden(x))
            x = self.dropout(x)
            x = leaky_relu(self.hidden2(x))
            x = self.out(x)
        return x


class Disc(nn.Module):
    def __init__(self,n_dim=3,l_dim=10,hidden=300,num_layers=3,num_heads=1,dropout=0.5,norm_first=False,cls=False,pair=False):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.l_dim
        self.embbed = nn.Linear(n_dim, l_dim)
        self.pair = pair
        if pair:
            self.pair_embed=PairEmbed(4, l_dim)
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=l_dim,nhead=num_heads,dim_feedforward=hidden,dropout=dropout,batch_first=True,
                activation=leaky,norm_first=norm_first),
                num_layers=num_layers,
            )
        if cls:
            self.cls=True
            self.encoder_cls =  nn.ModuleList([Block(embed_dim=l_dim, num_heads=num_heads,hidden=hidden,
                 dropout=dropout,)
                for i in range(2)])
        else:
            self.cls=False
        self.fc=cls
        if self.fc:
            self.hidden = nn.Linear(l_dim , 2 * hidden )
            self.hidden2 = nn.Linear(2 * hidden , hidden )
        self.out = nn.Linear(hidden , 1)

        self.cls_token = nn.Parameter(torch.zeros(1,1,l_dim),requires_grad=True)
        
    def forward(self, x, mask=None,noise=0,):
        if self.pair:
            attn_mask = self.pair_embed(v).view(-1, v.size(-1), v.size(-1))
        else:
            attn_mask=None
        if not self.cls:
            x = self.embbed(x)
            mask = torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device).bool()
            x = torch.concat((torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1)
            x = self.encoder(x, src_key_padding_mask=mask.bool())        
            x = x[:, 0, :]
            if self.fc:
                x = leaky_relu(self.hidden(x), 0.2)
                
                x = leaky_relu(self.hidden2(x), 0.2)
                
                x = self.out(x)
        else:
            x = self.embbed(x)
            x = self.encoder(x, src_key_padding_mask=mask.bool())

            cls_token=self.cls_token.expand(len(x),1,-1).to(x.device)
            #mask=torch.concat((torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1).to(x.device).bool()
            
            for layer in self.encoder_cls:
                cls_token=layer(x=x,x_cls=cls_token, padding_mask=mask)
            if self.fc:
                x = leaky_relu(self.hidden(cls_token), 0.2) 
                x = leaky_relu(self.hidden2(x), 0.2)
                x = self.out(x).reshape(-1)
            else:
                x = self.out(cls_token).reshape(-1)
        return x



