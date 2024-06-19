import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.T = config.block_size
        # self.register_buffer("bias", torch.tril(torch.ones(self.T, self.T))
        #                                 .view(1, 1, self.T, self.T))

    def forward(self, x):
        B, T, C = x.size() # (Batch_size, seq_len, embedding_dimensionality)
        head_dim = C // self.n_head
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # q: (B, T, C), k: (B, T, C), v: (B, T, C)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2) # (B, nh, T, hd)

        # # compute attention matrix
        # att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf')) # (B, nh, T, T)
        # att = F.softmax(att, dim=-1) # (B, nh, T, T)

        # # compute context vector
        # y = att @ v # (B, nh, T, hd)

        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B ,T, C)

        # output projection
        y = self.c_proj(y) # (B, T, C)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = std * (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"seq len should be less than equal to {T}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        x = pos_emb + tok_emb  # (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # x: (B, T, n_embd)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # x: (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups
        # Any params that is >=2D will be weight decayed -> matmuls, embeddings decay
        # otherwise no -> layernorm, all biases
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, 
            {'params': nodecay_params, 'weight_decay': 0.0}, 
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device_type
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ------------------------------------------------------------------------ #

# Simple command for single gpu
# python train.py
# For DDP launch
# torchrun --standalone --nproc_per_node 8 train.py


import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Setuo DDP
# torchrun command sets the new env variables
ddp = int(os.environ.get("RANK", -1)) != -1  # check for ddp run
if ddp:
    assert torch.cuda.is_available(), "CUDA required"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0   # for logging, checkpointing, etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"using device: {device}") 

device_type = "cuda" if device.startswith("cuda") else "cpu"

import tiktoken
class DataLoaderLite:
    def __init__(self,B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, 0.5M number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size should be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size} tokens')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)) # Random initialization for training from scratch
model.to(device)
model = torch.compile(model)

# wrap the model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


# train_loader = DataLoaderLite(B = 8, T = 1024)
train_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1. linear lr for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # 2. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3. in between use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# running tf32
torch.set_float32_matmul_precision("high")

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for step in range(5):
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps  # normalize since loss computation has reduction=mean over (B*T)
            loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # synchronize only on the last grad accm step (avg gradients incase of ddp)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    # tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    if master_process:
        print(f'step {step} | loss: {loss_accum.item()} | norm: {norm:.4f} | lr: {lr:.4e} |dt: {dt}ms | tok/sec: {tokens_per_sec}')

if ddp:
    destroy_process_group()


# model.eval()  # does not affect anything, since train and eval is same as no dropout, BN
# # prefix tokens
# num_return_sequences = 5
# max_length = 32
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, T)
# x = tokens.to(device)

# # generate, x: (B, T)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# print(x.size())
# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x) # (B, T, vocab_size)
#         # take the logits at the last position
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # top-k sampling of 50
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50)
#         # select a token from the topk probabilities
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         x = torch.cat((x, xcol), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)
