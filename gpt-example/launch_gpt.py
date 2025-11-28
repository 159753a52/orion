"""
Simplified launcher for GPT model with Orion scheduler.
Usage:
    LD_PRELOAD='/path/to/libinttemp.so' python launch_gpt.py --config gpt_config.json
"""

import argparse
import json
import threading
import time
from ctypes import cdll
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scheduler_frontend import PyScheduler

# GPT Model hyperparameters
emb_size = 128
head_size = 8
n_layer = 12
sequence_len = 128

torch.manual_seed(12046)


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


def block(backend_lib, it):
    backend_lib.block(it)


def check_stop(backend_lib):
    return backend_lib.stop()


def gpt_loop(
    model_name,
    batchsize,
    train,
    num_iters,
    rps,
    uniform,
    dummy_data,
    local_rank,
    barriers,
    client_barrier,
    tid,
    input_file=''
):
    """GPT inference loop compatible with Orion scheduler."""
    
    print(f"[GPT Client {tid}] Starting, batchsize={batchsize}, num_iters={num_iters}")
    
    # Load the interception library
    lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/cuda_capture/libinttemp.so"
    backend_lib = cdll.LoadLibrary(lib_path)
    
    # Wait for scheduler to be ready
    barriers[0].wait()
    print(f"[GPT Client {tid}] Passed initial barrier, building model...")
    
    # Create model
    device = f'cuda:{local_rank}'
    vocab_size = 256
    model = CharGPT(vocab_size).to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, vocab_size, (batchsize, sequence_len), device=device)
    
    print(f"[GPT Client {tid}] Model ready, starting inference loop...")
    
    batch_idx = 0
    start_time = time.time()
    
    # Barrier for backward setup (scheduler expects this after first iteration)
    barriers[0].wait()
    print(f"[GPT Client {tid}] Passed backward setup barrier")
    
    # Barrier for warmup (scheduler expects 10 warmup iters, we just sync here)
    barriers[0].wait()
    print(f"[GPT Client {tid}] Passed warmup barrier")
    
    while batch_idx < num_iters:
        iter_start = time.time()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Signal to scheduler that this iteration is done
        block(backend_lib, batch_idx)
        torch.cuda.synchronize()
        
        iter_time = time.time() - iter_start
        print(f"[GPT Client {tid}] Iteration {batch_idx} done, took {iter_time*1000:.2f} ms")
        
        batch_idx += 1
        
        # Check for stop signal from scheduler
        if check_stop(backend_lib):
            print(f"[GPT Client {tid}] Received stop signal")
            break
    
    # Final synchronization
    barriers[0].wait()
    print(f"[GPT Client {tid}] Passed final barrier")
    
    total_time = time.time() - start_time
    print(f"[GPT Client {tid}] Completed {batch_idx} iterations in {total_time:.2f} sec")
    print(f"[GPT Client {tid}] Throughput: {batch_idx/total_time:.2f} iter/sec")


def launch_jobs(config_dict_list, run_eval=True):
    """Launch GPT jobs with Orion scheduler."""
    
    print("=" * 50)
    print("GPT Orion Scheduler Launch")
    print("=" * 50)
    
    num_clients = len(config_dict_list)
    print(f"Number of clients: {num_clients}")
    
    # Initialize barriers
    num_barriers = num_clients + 1
    barriers = [threading.Barrier(num_barriers) for _ in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)
    
    # Load scheduler library
    sched_lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src/scheduler/scheduler_eval.so"
    print(f"Loading scheduler from: {sched_lib_path}")
    sched_lib = cdll.LoadLibrary(sched_lib_path)
    
    py_scheduler = PyScheduler(sched_lib, num_clients)
    
    # Extract configuration
    model_names = [cfg['arch'] for cfg in config_dict_list]
    model_files = [cfg['kernel_file'] for cfg in config_dict_list]
    additional_model_files = [cfg.get('additional_kernel_file') for cfg in config_dict_list]
    num_kernels = [cfg['num_kernels'] for cfg in config_dict_list]
    num_iters = [cfg['num_iters'] for cfg in config_dict_list]
    train_list = [cfg['args']['train'] for cfg in config_dict_list]
    additional_num_kernels = [cfg.get('additional_num_kernels') for cfg in config_dict_list]
    
    # Start client threads
    tids = []
    threads = []
    
    for i, cfg in enumerate(config_dict_list):
        model_args = cfg['args'].copy()
        model_args.update({
            "num_iters": num_iters[i],
            "local_rank": 0,
            "barriers": barriers,
            "client_barrier": client_barrier,
            "tid": i
        })
        
        thread = threading.Thread(target=gpt_loop, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)
    
    print(f"Client thread IDs: {tids}")
    
    # Start scheduler thread
    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,   # profile
            run_eval,
            False,  # reef
            False,  # sequential
            1,      # reef_depth / orion_max_be_duration
            1,      # hp_limit
            1,      # update_start
            train_list
        )
    )
    sched_thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    print("Client threads joined")
    
    sched_thread.join()
    print("Scheduler thread joined")
    
    print("=" * 50)
    print("All threads completed!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch GPT with Orion Scheduler')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()
    
    torch.cuda.set_device(0)
    
    with open(args.config) as f:
        config_dict = json.load(f)
    
    launch_jobs(config_dict)
