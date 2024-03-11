import torch
from torch import nn, optim
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
from torch import einsum

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_output, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_reduce = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_qs = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(n_head * d_k, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_output, bias=False)

        self.residual_transform = nn.Linear(d_model, d_output, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output, eps=1e-6)

    def forward(self, x, mask=None):
        q = self.w_reduce(x)
        k = q
        v = q
        #print("Shape of input tensor x:", x.shape)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        residual = self.residual_transform(residual)

        #print("Shape after w_reduce:", q.shape)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        #print("Shape of q after view:", q.shape)
        #print("Shape of k after view:", k.shape)
        #print("Shape of v after view:", v.shape)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class MultiHeadAttention_down(nn.Module):
    ''' Multi-Head Attention module with downsampling '''

    def __init__(self, n_head, d_model, d_k, d_v, d_output, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_reduce = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_qs = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(n_head * d_k, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(n_head * d_k, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_output, bias=False)

        self.residual_transform = nn.Linear(d_model, d_output, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output, eps=1e-6)
        self.downsample = nn.Linear(2 * d_output, d_output)  # Downsampling layer
        #self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x, mask=None):
        q = self.w_reduce(x)
        k = q
        v = q
        #print("Shape of input tensor x:", x.shape)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        fresidual = self.residual_transform(residual)


        #print("Shape after w_reduce:", q.shape)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        #print("Shape of q after view:", q.shape)
        #print("Shape of k after view:", k.shape)
        #print("Shape of v after view:", v.shape)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        # Downsampling
        sz_b, len_q, d_output = q.size()
        if len_q % 2 != 0:
            # Pad the sequence if it's odd-length
            padding = torch.zeros(sz_b, 1, d_output, device=q.device)
            q = torch.cat([q, padding], dim=1)
            len_q += 1  # Update length after padding

        q = q.view(sz_b, len_q // 2, 2 * d_output)  # Prepare for downsampling
        q = self.downsample(q)  # Apply downsampling
        '''# Prepare for max pooling
        q = q.transpose(1, 2)  # Shape [sz_b, d_output, len_q]

        # If sequence length is odd, pad by one column of zeros on the last dimension
        if q.size(2) % 2 != 0:
            padding = torch.zeros(sz_b, q.size(1), 1, device=q.device)
            q = torch.cat([q, padding], dim=2)

        # Apply max pooling
        q = self.maxpool(q)  # Shape [sz_b, d_output, len_q/2]

        # Transpose back to [sz_b, len_q/2, d_output]
        q = q.transpose(1, 2)'''

        return q

class CrossAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, d_output, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_output, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_output, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Linear projections for query, key, and value
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        residual = q
        q += residual

        q = self.layer_norm(q)

        return q

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class BidirectionalCrossAttention(nn.Module):
    '''
    Bidirectional attention, have two sequences attend to each other with 1 attention step
    designed to compute attention in two directions between two sequences
    '''
    def __init__(self, n_head, d_model, dim_head, m_model = None, dropout=0.1,prenorm = False):
        super().__init__()
        m_model = default(m_model, d_model)
        self.dim = d_model

        self.norm = nn.LayerNorm(d_model) if prenorm else nn.Identity()
        self.m_norm = nn.LayerNorm(m_model) if prenorm else nn.Identity()

        self.heads = n_head
        self.scale = dim_head ** -0.5


        self.dropout = nn.Dropout(dropout)
        self.m_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(d_model, dim_head * n_head, bias = False)
        self.m_to_qk = nn.Linear(m_model, dim_head * n_head, bias = False)

        self.to_v = nn.Linear(d_model, dim_head * n_head, bias = False)
        self.m_to_v = nn.Linear(m_model, dim_head * n_head, bias = False)

        self.to_out = nn.Linear(dim_head * n_head, d_model)
        self.m_to_out = nn.Linear(dim_head * n_head, m_model)

        '''self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()'''

    def forward(self,x,m,mask = None,m_mask = None,return_attn = False,rel_pos_bias = None):
        b, i, j, h, device = x.shape[0], x.shape[-2], m.shape[-2], self.heads, x.device

        x = self.norm(x)
        m = self.m_norm(m)

        # get shared query/keys and values for both
        qk, v = self.to_qk(x), self.to_v(x)
        m_qk, m_v = self.m_to_qk(m), self.m_to_v(m)

        # split out head
        qk, m_qk, v, m_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, m_qk, v, m_v))

        # get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, m_qk) * self.scale

        # relative positional bias, if supplied
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        '''# mask
        if exists(mask) or exists(m_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(m_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)'''

        # get attention along both x length and m length dimensions
        # shared similarity matrix
        attn = sim.softmax(dim = -1)
        m_attn = sim.softmax(dim = -2)

        # dropouts
        attn = self.dropout(attn)
        m_attn = self.m_dropout(m_attn)

        '''# talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)'''

        # src sequence aggregates values from m, m aggregates values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, m_v)
        m_out = einsum('b h j i, b h j d -> b h i d', m_attn, v)

        # merge heads and combine out
        out, m_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, m_out))

        out = self.to_out(out)
        m_out = self.m_to_out(m_out)

        #concatenate along the feature dimension
        combined_out = torch.cat((out, m_out), dim=1)

        # Map back to original dimension with an additional linear layer
        combined_dim = 256 #(1536
        final_out_linear = nn.Linear(combined_dim, self.dim).to(device)
        final_out = final_out_linear(combined_out)

        if return_attn:
            return final_out, attn, m_attn

        return final_out