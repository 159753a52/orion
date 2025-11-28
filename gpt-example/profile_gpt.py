"""
GPT Model Profiling Script for Orion

Usage:
    # For NCU profiling:
    ncu --set detailed --nvtx --nvtx-include "start/" -o output_ncu python profile_gpt.py --profile ncu
    
    # For NCU CSV output:
    ncu --csv --set detailed --nvtx --nvtx-include "start/" python profile_gpt.py --profile ncu > output_ncu.csv
    
    # For NSYS profiling:
    nsys profile -w true -t cuda,nvtx,osrt -s none -o output_nsys --cudabacktrace=true \
        --capture-range=cudaProfilerApi --stop-on-range-end=true -f true -x true \
        python profile_gpt.py --profile nsys
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(12046)

# Model hyperparameters
emb_size = 128
head_size = 8
n_layer = 12
sequence_len = 128


def attention(query, key, value, dropout, mask=None):
    B, T, C = query.shape
    scores = query @ key.transpose(-2, -1) / (C ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    w_att = dropout(F.softmax(scores, dim=-1))
    out = w_att @ value
    return out, w_att


class MaskedAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mask = self.tril[:T, :T]
        out, _ = attention(q, k, v, self.dropout, mask)
        return out


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        assert emb_size % head_size == 0
        n_head = emb_size // head_size
        heads = [MaskedAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.l1 = nn.Linear(emb_size, 4 * emb_size)
        self.l2 = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        out = self.dropout(self.l2(x))
        return out


class Block(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(emb_size, head_size)
        self.ff = FeedForward(emb_size)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        out = x + self.ff(self.ln2(x))
        return out


class CharGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        blocks = [Block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


def gpt_profile(batchsize, local_rank=0, do_eval=True, profile=None, warmup_iters=10, num_profile_iters=1, seq_len=512):
    """
    GPT model profiling function.
    
    Args:
        batchsize: batch size for inference
        local_rank: GPU device id
        do_eval: True for inference, False for training
        profile: 'ncu' for NCU profiling, 'nsys' for NSYS profiling, None for no profiling
        warmup_iters: number of warmup iterations
        num_profile_iters: number of profiled iterations
        seq_len: sequence length
    """
    global sequence_len
    sequence_len = seq_len
    print(f"GPT Profiling: batchsize={batchsize}, device={local_rank}, eval={do_eval}, profile={profile}, warmup={warmup_iters}, iters={num_profile_iters}, seq_len={seq_len}")
    
    device = f'cuda:{local_rank}'
    vocab_size = 256
    
    # Create model
    model = CharGPT(vocab_size).to(device)
    
    if do_eval:
        model.eval()
    else:
        model.train()
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batchsize, sequence_len), device=device)
    
    print(f"Starting {warmup_iters} warmup iterations...")
    
    # Warmup iterations
    for i in range(warmup_iters):
        if do_eval:
            with torch.no_grad():
                _ = model(dummy_input)
        else:
            _ = model(dummy_input)
        torch.cuda.synchronize()
    
    print(f"Warmup done. Starting {num_profile_iters} profiled iteration(s)...")
    
    batch_idx = 0
    
    while batch_idx < num_profile_iters:
        # Start profiling at first iteration
        if batch_idx == 0:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()
        
        start_time = time.time()
        
        if do_eval:
            with torch.no_grad():
                output = model(dummy_input)
        else:
            output = model(dummy_input)
        
        torch.cuda.synchronize()
        iter_time = time.time() - start_time
        
        # Stop profiling at last iteration
        if batch_idx == num_profile_iters - 1:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()
        
        print(f"Iteration {batch_idx}: {iter_time*1000:.2f} ms")
        batch_idx += 1
    
    print("Profiling done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT Model Profiling')
    parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('--profile', type=str, choices=['ncu', 'nsys', 'none'], default='none',
                        help='Profiling mode: ncu, nsys, or none')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--train', action='store_true', help='Run in training mode instead of eval')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=1, help='Number of profiled iterations')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    
    args = parser.parse_args()
    
    profile_mode = None if args.profile == 'none' else args.profile
    gpt_profile(
        batchsize=args.batchsize,
        local_rank=args.device,
        do_eval=not args.train,
        profile=profile_mode,
        warmup_iters=args.warmup,
        num_profile_iters=args.iters,
        seq_len=args.seq_len
    )
