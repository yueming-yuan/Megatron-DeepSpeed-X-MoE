# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from deepspeed.accelerator import get_accelerator
from torch.utils.checkpoint import detach_variable

from megatron import get_args
from megatron.memory import allocate_mem_buff
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from megatron.core.tensor_parallel.utils import (
    split_tensor_into_1d_equal_chunks,
    gather_split_1d_tensor,
)

from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, _set_cuda_rng_state

from megatron.core.utils import safely_set_viewless_tensor_data

import deepspeed

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'

# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None




class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """
    @staticmethod
    def forward(ctx, run_function, run_function_2, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations \
            = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = get_accelerator().get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)
            outputs = run_function_2(outputs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

        # HACK: currently when DeepSpeed is used, we always set
        # distribute_saved_activations to false, and use the following older
        # activation checkpointing mechanisms
        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            ctx.input_0_shape = args[0].data.shape
            args[0].data = split_tensor_into_1d_equal_chunks(args[0].data)
            args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(
                args[0].data)

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape))
        # HACK: currently when DeepSpeed is used, we always set
        # distribute_saved_activations to false, and use the following older
        # activation checkpointing mechanisms
        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            inputs[0].data = gather_split_1d_tensor(inputs[0].data)
            inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = get_accelerator().get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif len(outputs) == 2 and isinstance(outputs[1], torch.Tensor) and \
                torch.equal(outputs[1], torch.tensor(0).to(get_accelerator().device_name())):
            # a hacky solution to overcome issue when running old script examples/pretrain_gpt_distributed.sh
            outputs = (outputs[0],)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, None) + grads

def custom_checkpoint(function, function2, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    if deepspeed.checkpointing.is_configured():
        raise NotImplementedError("DeepSpeed checkpointing is not implemented for fine-grained control")
        # return deepspeed.checkpointing.checkpoint(function, *args)
    
    return CheckpointFunction.apply(function, function2,
                                    distribute_saved_activations, *args)