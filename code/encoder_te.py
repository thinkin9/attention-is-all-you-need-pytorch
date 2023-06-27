# Structure

# 1.RPC Setup
# 2.encoder via tvm
# 3.

########################################

# 1.RPC Setup
# ReferenceL Docs - VTA Tutorials - Optimize Tensor Operators - Matrix Multiply Blocking
# https://tvm.apache.org/docs/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-topic-vta-tutorials-optimize-matrix-multiply-opt-py

from __future__ import absolute_import, print_function

import os
import tvm
import tvm.testing
from tvm import te
import vta
import numpy as np

from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

# Load VTA parameters (3rdparty/vta-hw/config/vta_config.json file)
env = vta.get_env()

# PYNQ RPC IP, Port
host = os.environ.get("VTA_RPC_HOST", "192.168.1.152")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

assert tvm.runtime.enabled("rpc")
remote = rpc.connect(host, port)

# Reconfigure the JIT runtime
vta.reconfig_runtime(remote)

# Path of bitstream (Now, bitstream=None)
vta.program_fpga(remote, bitstream=None)

########################################

# 2.encoder via tvm
# ReferenceL Docs - VTA Tutorials - Optimize Tensor Operators - Matrix Multiply Blocking
# https://tvm.apache.org/docs/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-topic-vta-tutorials-optimize-matrix-multiply-opt-py

# import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm

# Hyperparams
d_model= 512
d_ff= 2048
n_head= 8
dk, dv = 64, 64
len_q = 30  # Set arbitrarily (between 20~40)

# Not used (We'll generate input randomly(or with autotuning))
# -> Don't need to consider batch size(If I'm right..)
sz_b = 256

qk_shape = (len_q, n_head*dk)
v_shape = (len_q, n_head*dv)  # Actually, qk_shape == v_shape
w_shape = (d_model, d_model)

qk_shape_head = (len_q, n_head, dk)
v_shape_head = (len_q, n_head, dv)  # Actually, qk_shape == v_shape

def Encoder(d_model, d_inner, n_head, dk, dv, dropout):
    
    # MultiHeadAttention:
    
    # tvm.te.placeholder(shape, dtype=None, name='placeholder')
    # I'm not sure of dtype
    q = te.placeholder(shape=qk_shape, name="q", dtype=env.inp_dtype)
    k = te.placeholder(shape=qk_shape, name="k", dtype=env.inp_dtype)
    v = te.placeholder(shape=v_shape, name="v", dtype=env.inp_dtype)
    
    # Residual (tvm.te doesn't have a function copy)
    # I think this is right
    #residual = te.compute(shape=qk_shape, lambda *i: q(*i))
    
    # tvm.te.compute(shape, fcompute, name='compute', tag='', attrs=None, varargs_names=None)
    # I'm not sure of dtype
    wq = te.placeholder(shape=w_shape, name="wq", dtype=env.wgt_dtype)
    wk = te.placeholder(shape=w_shape, name="wk", dtype=env.wgt_dtype)
    wv = te.placeholder(shape=w_shape, name="wv", dtype=env.wgt_dtype)
    
    # Computation order
    # C[i, :] = A[i:1]*B[1, :] + ... + A[i, l]*B[l,:] 
    # Operator Optimization on CPUs - 5. Matrix Multiplication - 5.3. Reordering Axes
    # https://tvm.d2l.ai/chapter_cpu_schedules/matmul.html
    q = te.compute(shape=qk_shape, lambda i, j: q[i, j] * wq[j, :], name="q")
    k = te.compute(shape=qk_shape, lambda i, j: k[i, j] * wk[j, :], name="k")
    v = te.compute(shape=v_shape, lambda i, j: v[i, j] * wv[j, :], name="v")

    # Encoder will be executed on VTA so that I think we need to optimize it in VTA-way (Data-tiling?)    
    # Docs - VTA Tutorials - Simple Matrix Multiply
    # https://tvm.apache.org/docs/topic/vta/tutorials/matrix_multiply.html#basic-mat-mult
    # Docs - VTA Tutorials - Optimize Tensor Operators - 2D Convolution Optimization
    # https://tvm.apache.org/docs/topic/vta/tutorials/optimize/convolution_opt.html#sphx-glr-topic-vta-tutorials-optimize-convolution-opt-py

    # View as Head
    # Pass through the pre-attention projection: len_q x (n_head*dk)
    # Separate different heads: len_q x n_head x dk
    q = te.compute(shape=qk_shape_head, lambda i, j, k: q[i, k + j * dk], name='q_reshape')
    k = te.compute(shape=qk_shape_head, lambda i, j, k: k[i, k + j * dk], name='k_reshape')
    v = te.compute(shape=v_shape_head, lambda i, j, k: v[i, k + j * dv], name='v_reshape')

    # ScaledDotProductAttention
    # => Relay 쓰는 게 조금 더 효율적일 거 같다...
