"""Three dense Qwen-style SwiGLU MLPs; reduced fixture, not full decoder.
Physical gate/up weight [2,F,D], exactly three layers. TP no-SP only.
No attention, norm, residual, bias, dropout, recompute, optimizer or loss.
Caller supplies a deterministic output gradient. No GPU launch at import.
"""
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import record_function

@dataclass(frozen=True)
class Config:
    tokens: int = 512
    dim: int = 2048
    hidden: int = 6144
    layers: int = 3
    tp: int = 2
    dp: int = 2
    seed: int = 20261117
    def validate(self):
        if self.layers != 3 or self.tp != 2 or self.dp != 2:
            raise ValueError('Fixed fixture: three layers, DP2 x TP2')
        if min(self.tokens,self.dim,self.hidden)<=0 or self.hidden % self.tp or self.dim % self.dp:
            raise ValueError('positive dimensions; hidden divisible by TP and dim divisible by DP required')


PHYSICAL_GPU_BY_RANK = (0,1,2,3)
TP_GROUPS = ((0,1),(2,3))
DP_GROUPS = ((0,2),(1,3))

def rank_layout(rank):
    if rank not in range(4):raise ValueError("rank must be 0..3")
    return PHYSICAL_GPU_BY_RANK[rank], rank//2, rank%2

def inputs(c, dtype=torch.float32):
    c.validate()
    g=torch.Generator(device='cpu').manual_seed(c.seed)
    weights=[]
    for _ in range(c.layers):
        weights.append((torch.randn(2,c.hidden,c.dim,generator=g,dtype=dtype)/(c.dim**0.5),
                        torch.randn(c.dim,c.hidden,generator=g,dtype=dtype)/(c.hidden**0.5)))
    xs=[torch.randn(c.tokens,c.dim,generator=g,dtype=dtype) for _ in range(c.dp)]
    dys=[torch.randn(c.tokens,c.dim,generator=g,dtype=dtype)/c.tokens for _ in range(c.dp)]
    return weights,xs,dys


def dense_step(x, weights, scopes=False):
    for i,(w13,w2) in enumerate(weights):
        ctx=lambda s:record_function(s) if scopes else nullcontext()
        with ctx(f'layer{i}.calc.gate_up'):
            gu=F.linear(x,w13.flatten(0,1)).unflatten(-1,(2,w13.shape[1]))
        with ctx(f'layer{i}.calc.swiglu'):
            h=F.silu(gu[...,0,:])*gu[...,1,:]
        with ctx(f'layer{i}.calc.down'):
            x=F.linear(h,w2)
    return x


def reference(c, dtype=torch.float32):
    weights,xs,dys=inputs(c,dtype)
    ws=[tuple(t.clone().requires_grad_() for t in pair) for pair in weights]
    outputs=[]; xgrads=[]
    for x,dy in zip(xs,dys):
        x=x.clone().requires_grad_(); y=dense_step(x,ws)
        y.backward(dy)
        outputs.append(y.detach()); xgrads.append(x.grad.detach())
    # FSDP mean over DP ranks; local input derivatives are not DP-averaged.
    grads=[tuple(t.grad.detach()/c.dp for t in pair) for pair in ws]
    return dict(config=asdict(c),weights=weights,xs=xs,dys=dys,outputs=outputs,xgrads=xgrads,wgrads=grads)

class CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group,layer):
        ctx.group,ctx.layer=group,layer
        return x
    @staticmethod
    def backward(ctx,dx):
        dx=dx.contiguous().clone()
        with record_function(f'layer{ctx.layer}.comm.tp.dx.all_reduce'):
            dist.all_reduce(dx,group=ctx.group)
        return dx,None,None

class ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group,layer):
        x=x.clone()
        with record_function(f'layer{layer}.comm.tp.y.all_reduce'):
            dist.all_reduce(x,group=group)
        return x
    @staticmethod
    def backward(ctx,dy):
        return dy,None,None

class GatedMLP(nn.Module):
    def __init__(self,pair,tp_rank,group,layer):
        super().__init__()
        w13,w2=pair; f=w13.shape[1]//2; s=slice(tp_rank*f,(tp_rank+1)*f)
        self.w13=nn.Parameter(w13[:,s,:].contiguous().clone())
        self.w2=nn.Parameter(w2[:,s].contiguous().clone())
        self.group,self.layer=group,layer
    def forward(self,x):
        x=CopyToTP.apply(x,self.group,self.layer)
        i=self.layer
        with record_function(f'layer{i}.calc.gate_up'):
            gu=F.linear(x,self.w13.flatten(0,1)).unflatten(-1,(2,self.w13.shape[1]))
        with record_function(f'layer{i}.calc.swiglu'):
            h=F.silu(gu[...,0,:])*gu[...,1,:]
        with record_function(f'layer{i}.calc.down'):
            y=F.linear(h,self.w2)
        return ReduceFromTP.apply(y,self.group,i)

class Stack(nn.Module):
    def __init__(self,layers):
        super().__init__();self.layers=nn.ModuleList(layers)
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            with record_function(f'layer{i}.forward'):x=layer(x)
        return x
