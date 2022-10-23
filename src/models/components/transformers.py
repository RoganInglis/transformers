import math
import torch
from torch import nn

import einops


def create_attn_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.d_model = dim_model

        period = 1 / 10000.0 ** (torch.arange(0, dim_model, 2).float() / dim_model)
        self.register_buffer('period', period)

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device).type_as(self.period)
        pos = pos.unsqueeze(1) * self.period.unsqueeze(0)
        pos = torch.stack((pos.sin(), pos.cos()), dim=-1).flatten(-2)
        return x + pos.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dropout=0.):
        """
        Computes the multi-head attention from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dropout: The dropout probability
        """
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.dim_inner = num_heads * dim_heads

        self.scale = dim_heads ** -0.5

        self.w_qkv = nn.Linear(in_features=dim_model, out_features=self.dim_inner)
        self.w_out = nn.Linear(in_features=self.dim_inner, out_features=dim_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        :param x_q: Tensor of shape (batch_size, seq_len_q, dim_model) containing the query vectors
        :param x_k: Tensor of shape (batch_size, seq_len_k, dim_model) containing the key vectors
        :param x_v: Tensor of shape (batch_size, seq_len_v, dim_model) containing the value vectors
        :param mask: Tensor of shape (batch_size, seq_len_q, seq_len_k) containing the mask for the attention
        :return:
        """
        # Project to query, key, value and split into heads
        qkv = einops.rearrange(
            self.w_qkv(torch.cat((x_q, x_k, x_v), dim=0)),
            'b s (h d) -> b h s d', h=self.num_heads
        )
        q, k, v = qkv.chunk(3, dim=0)

        # Calculate attention weights
        attn = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        w_attn = self.softmax(attn)
        w_attn = self.dropout(w_attn)

        # Calculate output
        out = torch.einsum('b h q k, b h k d -> b h q d', w_attn, v)
        out = einops.rearrange(out, 'b h s d -> b s (h d)')
        return self.w_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner=2048, dropout=0.):
        """
        Computes the feed-forward layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param dim_inner: The dimensionality of the inner layer
        :param dropout:
        """
        super().__init__()

        # TODO - it is possible to optimize this by removing the padding before and adding it after the linear layers
        # TODO - allow for different activation functions

        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, fn):
        """
        Computes the residual connection from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param fn: The function to apply the residual connection to
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim_model, eps=1e-5):
        """
        Computes the layer normalization from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param eps: The epsilon value for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim_model))
        self.beta = nn.Parameter(torch.zeros(dim_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, dropout=0.):
        """
        Computes the encoder layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dim_inner: The dimensionality of the inner feedforward layer
        :param dropout: The dropout probability
        """
        super().__init__()

        self.attn = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout))
        self.ff = Residual(FeedForward(dim_model, dim_inner, dropout))
        self.norm_1 = LayerNorm(dim_model)
        self.norm_2 = LayerNorm(dim_model)

    def forward(self, x, mask=None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model). Input to the encoder layer.
        :param mask: Tensor of shape (batch_size, seq_len, seq_len). Mask to be applied to the attention weights.
        :return:
        """
        x = self.attn(x, x, x, mask)
        x = self.norm_1(x)
        x = self.ff(x)
        return self.norm_2(x)


class DecoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, dropout=0.):
        """
        Computes the decoder layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dim_inner: The dimensionality of the inner feedforward layer
        :param dropout: The dropout probability
        """
        super().__init__()

        self.attn_1 = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout))
        self.attn_2 = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout))
        self.ff = Residual(FeedForward(dim_model, dim_inner, dropout))
        self.norm_1 = LayerNorm(dim_model)
        self.norm_2 = LayerNorm(dim_model)
        self.norm_3 = LayerNorm(dim_model)

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model). The input to the decoder.
        :param mem: Tensor of shape (batch_size, seq_len, dim_model). The memory from the encoder.
        :param src_mask: Tensor of shape (batch_size, 1, seq_len). The mask for the encoder.
        :param tgt_mask: Tensor of shape (batch_size, seq_len, seq_len). The mask for the decoder.
        :return:
        """
        x = self.attn_1(x, x, x, tgt_mask)
        x = self.norm_1(x)
        x = self.attn_2(x, mem, mem, src_mask)
        x = self.norm_2(x)
        x = self.ff(x)
        return self.norm_3(x)


class Encoder(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(dim_model, num_heads, dim_heads, dim_inner, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, num_heads, dim_heads, dim_inner, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, mem, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.):
        """
        Computes the transformer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dim_inner: The dimensionality of the inner feedforward layer
        :param num_layers: The number of encoder and decoder layers
        :param dropout: The dropout probability
        """
        super().__init__()
        self.encoder = Encoder(dim_model, num_heads, dim_heads, dim_inner, num_layers, dropout)
        self.decoder = Decoder(dim_model, num_heads, dim_heads, dim_inner, num_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        mem = self.encoder(src, src_mask)
        return self.decoder(tgt, mem, src_mask, tgt_mask)


class TransformerWrapper(nn.Module):
    def __init__(self, vocab_size, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim_model)
        self.pos_enc = PositionalEncoding(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            dim_model,
            num_heads,
            dim_heads,
            dim_inner,
            num_layers,
            dropout
        )
        self.out = nn.Linear(dim_model, vocab_size)
        self.out.weight = self.emb.weight

    def forward(self, src, tgt):
        src = self.dropout(self.pos_enc(self.emb(src) * math.sqrt(self.emb.embedding_dim)))
        tgt = self.dropout(self.pos_enc(self.emb(tgt) * math.sqrt(self.emb.embedding_dim)))
        return self.out(self.transformer(src, tgt)) * math.sqrt(self.emb.embedding_dim)
