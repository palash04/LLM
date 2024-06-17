from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.T = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(self.T, self.T))
                                        .view(1, 1, self.T, self.T))

    def forward(self, x):
        B, T, C = x.size() # (Batch_size, seq_len, embedding_dimensionality)
        head_dim = C // self.n_head
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self,n_embd, dim=2) # q: (B, T, C), k: (B, T, C), v: (B, T, C)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)

        # compute attention matrix
        att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf')) # (B, nh, T, T)
        att = F.softmax(att, dim=-1) # (B, nh, T, T)

        # compute context vector
        y = att @ v # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B ,T, C)

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50247
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # all the hidden layer blocks
            h = nn.ModuleList([ Block(config) for _ in range(config.n_layer) ]),
            # layer norm just before classifier layer
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # classifier layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" %model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M 
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


model = GPT.from_pretrained('gpt2')
print(model)
