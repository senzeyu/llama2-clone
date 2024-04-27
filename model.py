
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10_000.):
    # theta frequencies for rotary position embedding
    assert head_dim % 2 == 0, "Dimension of embeddings must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, step=2).float()
    theta = 1. / (theta ** (theta_numerator / head_dim)).to(device)
    
    # m corresponds to the position(s). same notation as the paper
    m = torch.arange(seq_len, device=device)
    
    freqs = torch.outer(m, theta).float() # shape: (seq_length, head_dim / 2)
    # torch.polar constructs a complex number arg1 is base, arg2 is the angle
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs) # create cos and sin on freqs
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, seq_length, num_heads, head_dim) -> (B, seq_length, num_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # (seq_length, head_dim / 2) -> (1, seq_length, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    x_rotated = x_complex * freqs_complex
    
    # (B, seq_length, num_head, head_dim/2) -> (B, seq_length, num_head, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class SelfAttention(nn.Module):
    """
    Multi-head Grouped-query Attention with KV Caching. This is for inference only
    """
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads
        # number of times K,V should be repeated to match the num_q_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (
                x.unsqueeze(3) # (B, seq_length, n_kv_heads, head_dim) -> # (B, seq_length, n_kv_heads, 1, head_dim)
                .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
            )
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """_summary_

        Args:
            start_pos (int): start position of the token that is being generate; this would start from 1

        Returns:
            _type_: _description_
        """
        batch_size, seq_len, _ = x.shape # (B, 1, dim)
        assert seq_len == 1, "Next token prediction. seq_len must be 1"
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # update cache 
        # e.g when we generate start_pos = 1, it attends to k,v from previous tokens, 0
        self.cache_k[:batch_size, start_pos-1:start_pos-1+seq_len] = xk
        self.cache_v[:batch_size, start_pos-1:start_pos-1+seq_len] = xv
        
        # retrieve cached k,v
        keys = self.cache_k[:batch_size, :start_pos-1+seq_len]
        values = self.cache_v[:batch_size, :start_pos-1+seq_len]
        
        # repeat the K, V multiple times to match Q
        keys = self.repeat_kv(keys, self.n_rep)
        values = self.repeat_kv(values, self.n_rep)
        
        # (B, 1, num_q_head, head_dim)
        # xq = xq.transpose(1, 2)
        # keys = keys.transpose(1, 2)
        # values = values.transpose(1, 2)
        # # (B, num_q_head, 1, head_dim) @ (B, num_q_head, head_dim, seq_len_kv) --> (B, num_q_head, 1, seq_len_kv)
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # output = torch.matmul(scores, values)
        # output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # i = 1 (cur token), j = seq_len_kv (previously generated tokens)
        scores = torch.einsum('b i h d, b j h d -> b h i j', xq, keys) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.einsum('b h i j, b j h d -> b h i d', scores, values)
        output = einops.rearrange(output, "b h i d -> b i (h d)")
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round-up the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        return self.w2(x)


class EncoderBlock(nn.Module):
    def __init__(self, args=ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        forward for inference only. 
        x is token of length 1
        """
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, args=ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set e.g from tokenizer."
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device
        )
        
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        llama forward with KV cache. `This is for inference only
        Args:
            tokens (troch.Tensor): input token, only one is needed because of KV caching
            start_pos (int): _description_
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one toekn at a time at inference w/ KV cach"
        h = self.tok_embeddings(tokens)
        
        # recomputing position encoding
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
           
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h)
        return output