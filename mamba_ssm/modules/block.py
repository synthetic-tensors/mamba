# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank):
    assert send_to_rank or receive_from_rank
    ops = []
    if send_to_rank is not None:
        ops.append(dist.P2POp(dist.isend, x, send_to_rank))
    if receive_from_rank is not None:
        ops.append(dist.P2POp(dist.irecv, receive_buffer, receive_from_rank))

    reqs = dist.batch_isend_irecv(ops)

    for req in reqs:
        req.wait()
    dist.barrier()

class SequenceParallelMixerFn(Function):
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

class SequenceParallelMixerLayer(nn.Module):
    def __init__(self, padding = 0):
        super(SequenceParallelMixerLayer, self).__init__()
        self.padding = padding
    def forward(self,x):
        return SequenceParallelMixerFn.apply(x, self.padding)

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
            context_parallel=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        if self.contextparallel:
            self.cpmixer = SequenceParallelMixerLayer(padding=mixer_cls.d_conv - 1)
        else:
            self.cpmixer = None

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        if self.cpmixer: #Context parallel - transfer some tokens to mix in with the conv layer to the next GPU
            hidden_states = self.cpmixer(hidden_states)

        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
