from collections import defaultdict
import pandas as pd
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.modules.block import Block
import torch
from functools import partial
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
#import torch.distributed.autograd as dist_autograd
#from einops import rearrange
if not dist.is_available():
    raise Exception("Distributed note abval")
import argparse

parser = argparse.ArgumentParser()
# This is always passed in by default
#parser.add_argument("--local_rank", type=int)
# These are your own arguments
#parser.add_argument("--master_addr", type=str)
parser.add_argument("--nproc_per_node", type=int)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--num_layers", type=int)
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus = args.nproc_per_node
num_layers = args.num_layers
batch = args.batch_size
d_model = 256
iterations = args.iterations
if num_gpus > 1:
    mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(num_gpus,))
#print(mesh_1d.get_group().bound_device_id)
#if dist.get_rank() == 0:
    #N.B. must use contiguous when splitting tensors for distributed ops!
    #sequence = rearrange(seq, 'i (n j) k -> i n j k', n = dist.get_world_size())
    #sequence = [sequence[:,i,:,:].contiguous() for i in range(dist.get_world_size())]
    #sequence = [seq[:,seq_per_gpu*x:seq_per_gpu*(x+1),:] for x in range(dist.get_world_size())]
    #sequence = [(torch.ones([batch,seq_per_gpu,256],dtype=torch.float32)*x).cuda() for x in range(dist.get_world_size())]
    #torch.save(sequence, f'sequence_{dist.get_rank()}.pt')
    #print('0',sequence[0].shape)
#else:
    #sequence = None
    
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

#print(dist.get_rank(), input_tensor.shape)
#dist.scatter(input_tensor, sequence, src=0)
#print(input_tensor[0,0,0], dist.get_rank())
if dist.is_initialized():
    world_size, rank = dist.get_world_size(), dist.get_rank()
else:
    world_size, rank = 1, 0

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

res_forward = list()
res_backward = list()
layers = []
for layer_idx in range(num_layers):
    block = Block(
            d_model,
            partial(Mamba2, context_parallel=dist.is_initialized(), sequence_parallel=False),
            norm_cls=RMSNorm,
            mlp_cls=nn.Identity,
            fused_add_norm=True,
            #context_parallel=True if dist.is_initialized() else False,
            #residual_in_fp32,
        )
    block.layer_idx = layer_idx
    layers.append(block)
model = nn.ModuleList(layers).cuda()
if rank == 0:
    print(model)

for s in range(10, 11):
    length = 2**s
    seq = torch.randn([iterations,batch,length*8,d_model],device='cpu')
    torch.save(seq,'seq.pt')
    #seq = torch.cat([(torch.ones([batch,length,256],dtype = torch.float32)*x).cuda() for x in range(num_gpus)], dim=1)
    assert seq.shape[1]%num_gpus == 0
    seq_per_gpu = seq.shape[2]//num_gpus
    print('running on ',rank, ' with ', seq_per_gpu)
    #Equal split sequences - easy test
    #sequence = rearrange(seq, 'i b (n j) k -> i n b j k', n = world_size)
    #sequence = [sequence[:,i,:,:].contiguous() for i in range(world_size)]
    #Split with padded repeats for 1d conv overlap
    #sequence = [seq[:, :, seq_per_gpu*r:seq_per_gpu*(r+1)+padding] for r in range(world_size)]
    sequence = [seq[:, :, seq_per_gpu * r:seq_per_gpu * (r + 1)] for r in range(world_size)] #Don't need padding with Mixer layer
    #with dist_autograd.context() as context_id:
    for i in range(iterations):
        #input_tensor = sequence[i,rank].cuda()
        #print(f"{sequence[rank].shape = }")
        batch = sequence[rank][i].clone().cuda().contiguous()
        torch.save(batch,f'input_{rank}.pt')
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start.record()
        residual = None
        for i,layer in enumerate(model):
            batch,residual = layer(batch,residual)
#            print(f'{i = } - {dist.get_rank() = } - {input_tensor.shape = }')
        end.record()
        torch.cuda.synchronize()
        r = torch.cuda.memory_reserved(rank)
        a = torch.cuda.memory_allocated(rank)
        t = start.elapsed_time(end)
        res_forward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
        if rank == 0:
            print("forward",s,i, a/10**9, r/10**9, 'GB')
            #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
            print("forward",s,i,t, 'ms')
        model.zero_grad()
        start.record()
        #dist_autograd.backward(context_id, [output[:,-1,:].sum()]) #For RPC only
        batch.sum().backward()
        end.record()
        torch.cuda.synchronize()
        r = torch.cuda.memory_reserved(rank)
        a = torch.cuda.memory_allocated(rank)
        t = start.elapsed_time(end)
        res_backward.append({'exp':s,'it':i,'res':r,'all':a,'time':t})
        if rank == 0:
            print("backward",rank,i, a/10**9, r/10**9, 'GB')
            #print(rank,prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=3))
            print("backward",rank,i,t, 'ms')
        if dist.is_initialized() and world_size > 1:
            dist.barrier()
    torch.save(batch, f"output_{rank}.pt")
    torch.save({x[0]:x[1].grad for x in model.named_parameters()}, f"grad_dict_{rank}.pt")
pd.DataFrame(res_forward).to_csv(f'res_fw_{rank}.csv')
pd.DataFrame(res_backward).to_csv(f'res_bw_{rank}.csv')
if dist.is_initialized():
    dist.destroy_process_group()

