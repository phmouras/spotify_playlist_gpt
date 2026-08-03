"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
from contextlib import nullcontext  # autocast for GPU
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group  # initialize and close communication w/ DDP

from model import GPTConfig, GPT  

parser = argparse.ArgumentParser()

parser.add_argument("--block_size", type=int, default = 128) # size of context
parser.add_argument("--n_layer", type=int, default = 4)      # transformer blocks: SA, FFN, LN, RC
parser.add_argument("--n_head", type=int, default = 4)       # head attention
parser.add_argument("--n_embd", type=int, default = 64)      # dim embedding. n_embd % n_head = 0
parser.add_argument("--checkpoint_dir", type=str, default = "./out/checkpoint_play.pt")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 1
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = True # if True, always save a checkpoint after each eval
always_save_checkpoint = False

# wandb logging, Weights and Biases
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

# data
# dataset = 'openwebtext'

# adamw optimizer: update weights during training
# learning_rate = 6e-4 # max learning rate
learning_rate = 1e-4 
# max_iters = 10000 # total number of training iterations
max_iters = 2000
weight_decay = 1e-1 # gradually reduces the magnitude of certain weights
beta1 = 0.9 
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# model init

with open("./corpus/vocab_size.txt") as f:
    vocab_size = int(f.read())

dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
model_args = dict(n_layer=args.n_layer, 
                  n_head=args.n_head, 
                  n_embd=args.n_embd, 
                  block_size=args.block_size,
                  bias=bias, 
                  vocab_size=vocab_size, 
                  dropout=dropout) # start with model_args from command line

gptconf = GPTConfig(**model_args) # model object

# gradient_accumulation_steps = 1 * 8 # used to simulate larger batch sizes
gradient_accumulation_steps = 2 
batch_size = 16 # n of sequences processed in parallel
block_size = gptconf.block_size

# model
n_layer = gptconf.n_layer
n_head = gptconf.n_head
n_embd = gptconf.n_embd

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
# lr_decay_iters = 1 # should be ~= max_iters per Chinchilla. Linear decay
lr_decay_iters = max_iters # For cossine decay: warmup_iters < lr_decay_iters  
# min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchill. This is for linear decay
min_lr = learning_rate / 10

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando em {device}")
print("PyTorch CUDA version:", torch.version.cuda)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# compile = True # use PyTorch 2.0 to compile the model to be faster
compile = False 
# -----------------------------------------------------------------------------

# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run? torchrun: Nao iremos utilizar no momento
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])  # global process identifier
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # local gpu used
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed (tokens)
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else: # python train.py
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:  
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset) # randomness control
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn. Speed up matrix multiplications 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) 

# poor man's data loader
# data_dir = os.path.join('data', dataset)

data_dir = "./corpus/corpus.bin"

data = np.memmap(data_dir, dtype=np.uint32,mode="r")

corpus_size = len(data)

split_size = 80
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    if split == 'train':
        inicio = 0
        fim = corpus_size * split_size // 100
    else: # val
        inicio = corpus_size * split_size // 100
        fim = corpus_size

    ix = torch.randint(fim - inicio - block_size, (batch_size,)) # randomize positions
    x = torch.stack([torch.from_numpy((data[inicio+i:inicio+i+block_size]).astype(np.int64)) for i in ix])    
    y = torch.stack([torch.from_numpy((data[inicio+i+1:inicio+i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

model = GPT(gptconf)
_ = model.to(device)

# AdamW optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type) # create adamw and save references

iter_num = 0
itecheckpoint = 0
best_val_loss = 1e9

if type(args.checkpoint_dir) != str or len(args.checkpoint_dir) == 0:
    saving = False
    print('new model wont be saved')
else:
    try: # load checkpoint
        cfg = torch.load(args.checkpoint_dir, map_location = device, weights_only=False) 
        model.load_state_dict(cfg['model'])
        optimizer.load_state_dict(cfg['optimizer']) # recover weights
        itecheckpoint = cfg["iter_num"]
        best_val_loss = cfg["best_val_loss"]
        print("Load Checkpoint Sucessfull")
    except Exception as e:
        print(f"\nnew model checkpoint will be saved as: {args.checkpoint_dir} \n: {e}")
    saving = True


# checkpoint_load_path = args.checkpoint_dir
# checkpoint_filename = os.path.basename(checkpoint_load_path)
# checkpoint_save_path = os.path.join(args.path_out,checkpoint_filename)

    # # fix the keys of the state dictionary :(
    # # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # model.load_state_dict(state_dict)
    
iter_num = itecheckpoint

# elif init_from.startswith('gpt2'):
#     print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
#     # initialize from OpenAI GPT-2 weights
#     override_args = dict(dropout=dropout)
#     model = GPT.from_pretrained(init_from, override_args)
#     # read off the created config params, so we can store them into checkpoint correctly
#     for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#         model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value
# model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model. Nao usaremos
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container: Nao usaremos
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad() # no backward for validation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:  # evaluate the 2 sets
        losses = torch.zeros(eval_iters) # create vector w/ eval_iters components
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() 
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup). Lr -> Step size
def get_lr(it):  # Variable lr in time
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff decay 1..0
    return min_lr + coeff * (learning_rate - min_lr) # learnin rate between lr and min_lr

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=gptconf)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed. Use model
running_mfu = -1.0

print("X shape:", X.shape) # (batch_size, block_size)
print("Y shape:", Y.shape)
print("Token mínimo:", X.min().item())
print("Token máximo:", X.max().item())

print("X primeira sequência:")
print(X[0, :20].tolist())
print("Y primeira sequência:")
print(Y[0, :20].tolist())

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss() 
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss and saving:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(), # trained weights
                    'optimizer': optimizer.state_dict(), # adamw state
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    # 'config': config,
                    'gptconfig': gptconf
                }
                print(f"saving checkpoint to {args.checkpoint_dir}")
                torch.save(checkpoint, args.checkpoint_dir)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps): 
        if ddp: # Not in use
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y) # calculate loss
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward() # calculate gradiants, which indicate the direction in which each weight should change to reduce the loss
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer) # update parameters w/ mixed precision
    scaler.update() # fit scale factor dinamically
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True) # clean gradients

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps # recover original scale
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt) # mfu -> fracao da capacidade computacional gpu que esta sendo utilizada
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
