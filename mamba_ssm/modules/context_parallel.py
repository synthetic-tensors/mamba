from typing import Optional

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    assert send_to_rank is not None or receive_from_rank is not None
    ops = []
    if send_to_rank is not None:
        ops.append(dist.P2POp(dist.isend, x, send_to_rank))
    if receive_from_rank is not None:
        ops.append(dist.P2POp(dist.irecv, receive_buffer, receive_from_rank))

    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    dist.barrier()

class ContextParallelMixerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, padding=0):
        #Prepends the last n_padding tokens from layer_n to layer_{n+1}
        #These are mixed into subsequent tokens of layer n+1 by convolution, but their index is then discarded
        # the convolution is causal, so the mixing only goes in one direction
        rank, world_size = dist.get_rank(), dist.get_world_size()
        ctx.padding = padding
        if world_size == 1:
            return x

        send_to_rank = rank + 1 if rank < world_size - 1 else None
        receive_from_rank = rank - 1 if rank > 0 else None
        #print('dist', rank, send_to_rank, receive_from_rank)
        #_, pre_tokens = x.split(x.shape[1]-self.padding, dim=1)
        pre_tokens = x[:,-ctx.padding:].contiguous()
        #print('dist',rank,pre_tokens.requires_grad)
        assert pre_tokens.shape[1] == ctx.padding
        receive_buffer = torch.zeros_like(pre_tokens, requires_grad=True).contiguous() #TODO this isn't used by rank=0
        send_and_receive_(pre_tokens, receive_buffer, send_to_rank, receive_from_rank)
        if rank > 0:
            x = F.pad(x, (0, 0, ctx.padding, 0), 'constant', 0)
            x[:,:ctx.padding] = receive_buffer
            #print('dist',rank,'receive_buffer grad',receive_buffer.requires_grad)
        #print('x', rank, x.shape)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        """
        grad x is input with the padding tokens from the next layer
        the input of forward is not padded, this gradient needs to be popped and transfered
        to the previous layer...
        """
        rank, world_size = dist.get_rank(), dist.get_world_size()
        #print('grad_x', rank, grad_x.shape)
        if world_size == 1:
            return grad_x, None
        send_to_rank = rank -1 if rank > 0 else None
        receive_from_rank = rank + 1 if rank < world_size - 1 else None
        pre_tokens_grad = grad_x[:,:ctx.padding].contiguous()
        if rank > 0:
            grad_x_out = grad_x[:,ctx.padding:].contiguous()
        else:
            grad_x_out = grad_x.clone()
        assert pre_tokens_grad.shape[1] == ctx.padding
        receive_buffer = torch.zeros_like(pre_tokens_grad).contiguous() #TODO this isn't used by rank=0
        send_and_receive_(pre_tokens_grad, receive_buffer, send_to_rank, receive_from_rank)
        if rank < world_size -1:
            grad_x_out[:,-ctx.padding:] += receive_buffer
        return grad_x_out, None

class ContextParallelMixerLayer(nn.Module):
    def __init__(self, padding = 0):
        super(ContextParallelMixerLayer, self).__init__()
        self.padding = padding
    def forward(self,x):
        return ContextParallelMixerFn.apply(x, self.padding)