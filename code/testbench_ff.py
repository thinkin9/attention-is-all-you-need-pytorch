import argparse, json, os, requests, sys, time
import numpy as np

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Load VTA parameters (3rdparty/vta-hw/config/vta_config.json file)
env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
# device="arm_cpu"
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu
print("env.target: ", target)

#if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    #tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    #racker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
device_host = os.environ.get("VTA_RPC_HOST", "192.168.1.152")
device_port = os.environ.get("VTA_RPC_PORT", "9091")
    #if not tracker_host or not tracker_port:
remote = rpc.connect(device_host, int(device_port))
    #else:
    #    remote = autotvm.measure.request_remote(
    #        env.TARGET, tracker_host, int(tracker_port), timeout=10000
    #    )

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
reconfig_start = time.time()
vta.reconfig_runtime(remote)
vta.program_fpga(remote, bitstream=None)
reconfig_time = time.time() - reconfig_start
print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# # In simulation mode, host the RPC server locally.
# else:
#     remote = rpc.LocalSession()

#     if env.TARGET in ["intelfocl"]:
#         # program intelfocl aocx
#         vta.program_fpga(remote, bitstream="vta.bitstream")

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

# Hyperparams
d_model= 512
d_ff= 2048
n_head= 8
dk, dv = 64, 64
len_q = 30  # Set arbitrarily (between 20~40)

dropout_rate = 0.1
epsilon_rate = 1e-6

# Not used (We'll generate input randomly(Just check for inference time/gops/ ...))
# -> Don't need to consider batch size(If I'm right..)
sz_b = 256

qk_shape = (len_q, n_head*dk)
v_shape = (len_q, n_head*dv)  # Actually, qk_shape == v_shape
fc_shape = (n_head*dv, d_model)
w_shape = (d_model, d_model)
wff1_shape = (d_model, d_ff)
wff2_shape = (d_ff, d_model)

qk_shape_head = (len_q, n_head, -4)
v_shape_head = (len_q, n_head, -4)  # Actually, qk_shape == v_shape

def Encoder(q_input, w1, w2, gamma, beta):
    
    # MultiHeadAttention:
    # FeedForward:
    
    # MultiHeadAttention:
    # 1. qkv
    # 2. residual
    # 3. weights
    # 4. nn.Linear
    # 5. View as head
    # 6. ScaledDotProductAttention
        # 6.1. matmul:    Q x K.T
        # 6.2. scale:     sqrt(dk)
        # 6.3. mask:      Not implemented (Mask only for Decoder)
        # 6.4. Softmax:   relay.nn.softmax
        # 6.5. matmul:    attn x V
    # 7. concate
    # 8. Linear
    # 9. dropout
    # 10. residual-sum
    # 11. layer-norm
    
    
    # FeedForward:
    # 1. residual
    # 2. weights
    # 3. nn.Linear
    # 4. relu
    # 5. nn.Linear
    # 6. dropout
    # 7. residual-sum
    # 8. layer-norm

    
    # 1. qkv
    # tvm.relay.var(name_hint, type_annotation=None, shape=None, dtype='float32', span=None)
    # I'm not sure of dtype
    # q = relay.var(name_hint="q", shape=qk_shape, dtype=env.inp_dtype)
    # k = relay.var(name_hint="k", shape=qk_shape, dtype=env.inp_dtype)
    # v = relay.var(name_hint="v", shape=v_shape, dtype=env.inp_dtype)
    # q = q_input
    # k = relay.copy(q)
    # v = relay.copy(q)
    
    # # 2. residual
    # # tvm.relay.copy(data)
    # residual = relay.copy(q)
    
    # # 3. weights
    # # I'm not sure of dtype
    # # tvm.relay.var(name_hint, type_annotation=None, shape=None, dtype='float32', span=None)
    # # wq = relay.var(name_hint="wq", shape=w_shape, dtype=env.wgt_dtype)
    # # wk = relay.var(name_hint="wk", shape=w_shape, dtype=env.wgt_dtype)
    # # wv = relay.var(name_hint="wv", shape=w_shape, dtype=env.wgt_dtype)
    
    # # 4. nn.Linear
    # # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    # # I'm not sure of dtype
    # q = relay.nn.dense(data=q, weight=wq)
    # k = relay.nn.dense(data=k, weight=wk)
    # v = relay.nn.dense(data=v, weight=wv)
    
    # # 5. View as head
    # # Pass through the pre-attention projection: len_q x (n_head*dk)
    # # Separate different heads: len_q x n_head x dk
    # # Transpose for attention dot product: n_head x len_q x dv
    
    # # tvm.relay.reshape(data, newshape, allowzero=False)
    # # tvm.relay.transpose(data, axes=None)
    # q_heads = relay.transpose(data=relay.reshape(data=q, newshape=qk_shape_head), axes=(1, 0, 2))
    # k_heads = relay.transpose(data=relay.reshape(data=k, newshape=qk_shape_head), axes=(1, 0, 2))
    # v_heads = relay.transpose(data=relay.reshape(data=v, newshape=v_shape_head), axes=(1, 0, 2))


    # # 6. ScaledDotProductAttention
    
    # # 6.1. matmul:    Q x K.T
    # # tvm.relay.nn.matmul(tensor_a, tensor_b, units=None, out_dtype='', transpose_a=False, transpose_b=False)
    # # tvm.relay.nn.batch_matmul(tensor_a, tensor_b, out_dtype='', transpose_a=False, transpose_b=True)
    # attn = relay.nn.batch_matmul(tensor_a=q_heads, tensor_b=relay.transpose(data=k_heads, axes=(0, 2, 1)))
    
    # # 6.2. scale:     sqrt(dk)
    # # sqrt(64) = 8
    # # tvm.relay.divide(lhs, rhs)
    # scale_coef = relay.const(value = 1 / 8)
    # attn = relay.multiply(q, scale_coef)
    # #attn = relay.divide(lhs=attn, rhs=dk ** 0.5)
    
    # # 6.3. mask:      Not implemented (Mask only for Decoder)
    
    # # 6.4. Softmax:   relay.nn.softmax
    # # tvm.relay.nn.softmax(data, axis=- 1)
    # attn = relay.nn.softmax(data=attn, axis=-1)
    
    # # 6.5. matmul:    attn x V
    # # tvm.relay.nn.matmul(tensor_a, tensor_b, units=None, out_dtype='', transpose_a=False, transpose_b=False)
    # # tvm.relay.nn.batch_matmul(tensor_a, tensor_b, out_dtype='', transpose_a=False, transpose_b=True)
    # output = relay.nn.batch_matmul(tensor_a=attn, tensor_b=v_heads)
    
    # # 7. concate
    # # output: n_head x len_q x dv (current) -> len_q x n_head x dv -> len_q x (n_head*dk)
    # # tvm.relay.transpose(data, axes=None)
    # # tvm.relay.reshape(data, newshape, allowzero=False)
    # q = relay.reshape(data=relay.transpose(data=output, axes=(1, 0, 2)), newshape=qk_shape)
    
    # # 8. Linear
    # # tvm.relay.var(name_hint, type_annotation=None, shape=None, dtype='float32', span=None)
    # # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    # #fc = relay.var(name_hint="fc", shape=fc_shape, dtype=env.wgt_dtype)
    # q = relay.nn.dense(data=q, weight=fc)
    
    # # 9. dropout
    # # tvm.relay.nn.dropout(data, rate=0.5)
    # q = relay.nn.dropout(data=q, rate=dropout_rate)
    
    # # 10. residual-sum
    # # tvm.relay.add(lhs, rhs)
    # q = relay.add(lhs=q, rhs=residual)
    
    # # 11. layer-norm
    # # tvm.relay.nn.layer_norm(data, gamma, beta, axis=- 1, epsilon=1e-05, center=True, scale=True)¶
    # gamma_coef = relay.const(value = 1)
    # beta_coef = relay.const(value = 0)
    # q = relay.nn.layer_norm(data=q, gamma=gamma_coef, beta=beta_coef, axis=-1, epsilon=epsilon_rate)
    
    
    # FeedForward:
    q = q_input
    residual = relay.copy(q)
    
    #q1 = relay.nn.dense(data=q, weight=w1)
    q1 = relay.nn.matmul(tensor_a=q, tensor_b=w1)
    
    q2 = relay.nn.relu(data=q1)
    
    #q3 = relay.nn.dense(data=q2, weight=w2)
    q3 = relay.nn.matmul(tensor_a=q2, tensor_b=w2)
    
    q4 = relay.nn.dropout(data=q3, rate=dropout_rate)
    
    q5 = relay.add(lhs=q4, rhs=residual) 
    
    q5_cast = relay.cast(q5, dtype="float32")
    q6 = relay.nn.layer_norm(data=q5_cast, gamma=gamma, beta=beta, axis=-1, epsilon=epsilon_rate)
    return q5


q_input = relay.var(name_hint="q", shape=qk_shape, dtype=env.inp_dtype)

w1 = relay.var(name_hint="w1", shape=wff1_shape, dtype=env.wgt_dtype)
w2 = relay.var(name_hint="w2", shape=wff2_shape, dtype=env.wgt_dtype)

gamma = relay.var(name_hint="gamma", shape=(d_model,), dtype="float32")
beta = relay.var(name_hint="beta", shape=(d_model,), dtype="float32")
q_output = Encoder(q_input, w1, w2, gamma, beta)

# Measure build start time
build_start = time.time()

fn_encoder = relay.Function([q_input, w1, w2, gamma, beta], q_output)
print(fn_encoder)
## 0628 14:54:여기까지는 잘됨

# Front end compilation
fmod = tvm.IRModule.from_expr(fn_encoder)
print(fmod)
## 0628 14:56:여기까지는 잘됨

relay_prog = fmod["main"]
print(relay_prog)
## relay_prog == fn_encoder
## 0628 15:03:여기까지는 잘됨

# Option1
graph, lib, params = relay.build(fn_encoder, target=tvm.target.Target(target, host=env.target_host))
print(graph)
print(lib)
print(params)
#graph, lib, params = relay.build(fn_encoder, target=tvm.target.Target(env.target if device == "vta" else env.target_vta_cpu, host=env.target_host))
## 0628 15:08:여기까지는 잘됨

# Option2
# with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
#     graph, lib, params = relay.build(fn_encoder, target=tvm.target.Target(target, host=env.target_host))
#     print(graph)
#     print(lib)
#     print(params)
    
## No difference between Option 1 and 2
build_time = time.time() - build_start
print("Encoder" + " inference graph built in {0:.2f}s!".format(build_time))
## Encoder Inference graph build in 0.14s
## 0628 15:11:여기까지는 잘됨

# Send the inference library over to the remote RPC server
temp = utils.tempdir()
lib.export_library(temp.relpath("graphlib.tar"))
remote.upload(temp.relpath("graphlib.tar"))
lib = remote.load_module("graphlib.tar")
## 0628 15:13: 오류 X

# if env.TARGET == "intelfocl":
#     ctxes = [remote.ext_dev(0), remote.cpu(0)]
#     m = graph_executor.create(graph, lib, ctxes)
#     오류 O
# else:
#     # Graph runtime
m = graph_executor.create(graph, lib, ctx)