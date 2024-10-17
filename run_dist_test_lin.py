from collections import defaultdict
import pandas as pd
from mamba_ssm import Mamba2
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

#import torch.distributed.autograd as dist_autograd
#from einops import rearrange
if not dist.is_available():
    raise Exception("Distributed note abval")
import argparse

class TestLayer(nn.Module):
    def __init__(self, d_model, tag=None):
        super(TestLayer, self).__init__()
        self.tag = tag
        self.model = nn.Linear(d_model,d_model)
    def forward(self, x):
        print(f"Running {self.tag} on {dist.get_rank()}")
        return self.model(x)


class TestModel(nn.Module):
    def __init__(self, num_layers, d_model):
        super(TestModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            test_layer = TestLayer(d_model, tag = str(i))
            self.layers.append(test_layer)
        
        #model = nn.Sequential(*layers).to(rank)
    def forward(self, x):
        #print(f"Running {self.tag} on {dist.get_rank()}")
        for layer in self.layers:
            x = layer(x)
        return x

parser = argparse.ArgumentParser()
# This is always passed in by default
#parser.add_argument("--local_rank", type=int)
# These are your own arguments
#parser.add_argument("--master_addr", type=str)
parser.add_argument("--fsdp", type=int)
parser.add_argument("--nproc_per_node", type=int)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--iterations", type=int)
parser.add_argument("--num_layers", type=int)
parser.add_argument("--d_model", type=int)
args = parser.parse_args()
print(args)
torch.manual_seed(args.random_seed)
num_gpus = args.nproc_per_node
num_layers = args.num_layers
batch = args.batch_size
iterations = args.iterations
d_model = args.d_model
dist.init_process_group("nccl")
#mesh_1d = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(num_gpus,))
#device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(args.fsdp, num_gpus//args.fsdp), mesh_dim_names=('dp','cp'))
#cp_mesh, dp_mesh = device_mesh['cp'], device_mesh['dp']
#print(dist.get_rank(),cp_mesh, dist.get_process_group_ranks(cp_mesh.get_group()))
#print(dist.get_rank(),dp_mesh, dist.get_process_group_ranks(dp_mesh.get_group()))
#print(dist.get_rank(), device_mesh.get_group(mesh_dim='dp'),device_mesh.get_group(mesh_dim='cp'))
#print(dist.get_group())
#print(dist.get_rank(group=dist.get_group))
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
world_size, rank = dist.get_world_size(), dist.get_rank()
torch.cuda.set_device(rank)

model = TestModel(num_layers, d_model).to(rank)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

res_forward = list()
res_backward = list()
# Init FSDP using the dp device mesh
model_wrap_policy = ModuleWrapPolicy([TestLayer])
sharded_model = FSDP(model, auto_wrap_policy=model_wrap_policy)#use_orig_params=True)
#print(next(sharded_model.parameters()).device)
print(f"Rank {rank}", sharded_model)
print(f"Rank {rank}", model_wrap_policy)
sharded_model = sharded_model.train()
for s in range(21,22):
    length = 2**s
    seq = torch.randn([iterations,batch,length,d_model],device='cpu')
    #torch.save(seq,'seq.pt')
    #seq = torch.cat([(torch.ones([batch,length,256],dtype = torch.float32)*x).cuda() for x in range(num_gpus)], dim=1)
    assert seq.shape[2]%num_gpus == 0, "Not the right sequence shape"
    seq_per_gpu = seq.shape[2]//num_gpus
    if rank ==0:
        print("Running ", s, " with ", seq_per_gpu, " per gpu")
    #print('running on ',dist.get_rank(), ' with ', seq_per_gpu)
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
        input_tensor = sequence[rank][i].to(rank).contiguous()
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        start.record()
        output = sharded_model(input_tensor)
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
        output.sum().backward()
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
        dist.barrier()
    torch.save(input_tensor,f'input_{rank}.pt')
    torch.save(output, f"output_{rank}.pt")
    torch.save({x[0]:x[1].grad for x in model.named_parameters()}, f"grad_dict_{rank}.pt")
pd.DataFrame(res_forward).to_csv(f'res_fw_{rank}.csv')
pd.DataFrame(res_backward).to_csv(f'res_bw_{rank}.csv')
dist.destroy_process_group()

exit()

if dist.get_world_size() > 1:
    tensor_list = gather(dist.get_rank(),output)
    #print(dist.get_rank(), [x[0,0,0] for x in tensor_list])

    if dist.get_rank() == 0:
        torch.save(tensor_list, f'output.pt')
else:
    torch.save(output, f'output.pt')


def gather(rank, tensor):
        #group = dist.new_group(list(range(rank + 1)))
        #shape = tensor.shape
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor) #  group=group)
        return tensor_list

#input_tensor = torch.zeros([batch,seq_per_gpu,256], device='cuda')
