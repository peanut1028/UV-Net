import torch
from torch import nn
from torch.nn.init import xavier_normal_, constant_
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from xformers.ops import memory_efficient_attention
from flash_attn.ops.activations import swiglu
import math


def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout_p=0.0, xformers_or_torch_attn="xformers"):
        super(MAB, self).__init__()

        self.xformers_or_torch_attn = xformers_or_torch_attn

        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.fc_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.fc_k = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_v = nn.Linear(dim_K, dim_V, bias=True)
        self.fc_o = nn.Linear(dim_V, dim_V, bias=True)

        # NOTE: xavier_uniform_ might work better for a few datasets
        xavier_normal_(self.fc_q.weight)
        xavier_normal_(self.fc_k.weight)
        xavier_normal_(self.fc_v.weight)
        xavier_normal_(self.fc_o.weight)

        # NOTE: this constant bias might work better for a few datasets
        # constant_(self.fc_q.bias, 0.01)
        # constant_(self.fc_k.bias, 0.01)
        # constant_(self.fc_v.bias, 0.01)
        # constant_(self.fc_o.bias, 0.01)

        # NOTE: this additional LN for queries/keys might be useful for some
        # datasets (currently it looks like DOCKSTRING)
        # It is similar to this paper https://arxiv.org/pdf/2302.05442
        # and https://github.com/lucidrains/x-transformers?tab=readme-ov-file#qk-rmsnorm

        # self.ln_q = nn.LayerNorm(dim_Q, eps=1e-8)
        # self.ln_k = nn.LayerNorm(dim_K, eps=1e-8)


    def forward(self, Q, K, adj_mask=None):
        batch_size = Q.size(0)
        E_total = self.dim_V
        assert E_total % self.num_heads == 0, "Embedding dim is not divisible by nheads"
        head_dim = E_total // self.num_heads

        Q = self.fc_q(Q)
        V = self.fc_v(K)
        K = self.fc_k(K)

        # Additional normalisation for queries/keys. See above
        # Q = self.ln_q(Q).to(torch.bfloat16)
        # K = self.ln_k(K).to(torch.bfloat16)

        Q = Q.view(batch_size, -1, self.num_heads, head_dim)
        K = K.view(batch_size, -1, self.num_heads, head_dim)
        V = V.view(batch_size, -1, self.num_heads, head_dim)

        if self.xformers_or_torch_attn in ["torch"]:
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)


        if adj_mask is not None:
            adj_mask = adj_mask.expand(-1, self.num_heads, -1, -1)

        if self.xformers_or_torch_attn == "xformers":
            out = memory_efficient_attention(Q, K, V, attn_bias=adj_mask, p=self.dropout_p if self.training else 0)
            out = out.reshape(batch_size, -1, self.num_heads * head_dim)
            
        elif self.xformers_or_torch_attn in ["torch"]:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=adj_mask, dropout_p=self.dropout_p if self.training else 0, is_causal=False
                )
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)


        out = out + F.mish(self.fc_o(out))

        return out


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout, xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(X, X, adj_mask=adj_mask)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout, xformers_or_torch_attn):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, dropout_p=dropout, xformers_or_torch_attn=xformers_or_torch_attn)

    def forward(self, X, adj_mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, adj_mask=adj_mask)



class SmallMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(nn.Linear(in_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                elif i != num_layers - 1:
                    self.mlp.append(nn.Linear(inter_dim, inter_dim))
                    if use_ln:
                        self.mlp.append(nn.LayerNorm(inter_dim))
                    self.mlp.append(nn.Mish())
                else:
                    self.mlp.append(nn.Linear(inter_dim, out_dim))

                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

            self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)
    

class GatedMLPSingle(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        use_ln=False,
    ):
        super().__init__()

        # Uncomment if you want dropout here
        # self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, 2 * inter_dim, bias=True)
        self.fc2 = nn.Linear(inter_dim, out_dim, bias=True)
        self.use_ln = use_ln

        if self.use_ln:
            self.ln = nn.LayerNorm(2 * inter_dim, eps=1e-8)

        # if dropout_p > 0:
        #     self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        if self.use_ln:
            y = self.ln(self.fc1(x))
        else:
            y = self.fc1(x)

        y, gate = y.chunk(2, dim=-1)
        y = swiglu(gate, y)

        # if self.dropout_p > 0:
        #     y = self.dropout(y)
        y = self.fc2(y)
        
        return y


class GatedMLPMulti(nn.Module):
    def __init__(
        self,
        in_dim,
        inter_dim,
        out_dim,
        dropout_p=0.0,
        num_layers=2,
        use_ln=False,
    ):
        super().__init__()

        self.mlp = []

        if num_layers == 1:
            self.mlp = nn.Sequential(
                GatedMLPSingle(in_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=False),
                nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Mish(),
            )
        else:
            for i in range(num_layers):
                if i == 0:
                    self.mlp.append(GatedMLPSingle(in_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                elif i != num_layers - 1:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, inter_dim, dropout_p=dropout_p, use_ln=use_ln))
                else:
                    self.mlp.append(GatedMLPSingle(inter_dim, inter_dim, out_dim, dropout_p=dropout_p, use_ln=use_ln))
                
                if dropout_p > 0:
                    self.mlp.append(nn.Dropout(p=dropout_p))

                self.mlp.append(nn.Mish())

        self.mlp = nn.Sequential(*self.mlp)


    def forward(self, x):
        return self.mlp(x)