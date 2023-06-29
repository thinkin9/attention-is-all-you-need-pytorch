# FeedForward on VTA

# Load VTA parameters

# Load VTA parameters (3rdparty/vta-hw/config/vta_config.json file)
import vta
env = vta.get_env()

import time, os
import tvm
from tvm import rpc
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform
from vta.testing import simulator
assert tvm.runtime.enabled("rpc")

pynq_host = os.environ.get("VTA_RPC_HOST", "192.168.1.152")
pynq_port = os.environ.get("VTA_RPC_PORT", "9091")

remote = rpc.connect(pynq_host, int(pynq_port))

reconfig_start = time.time()
vta.reconfig_runtime(remote)
vta.program_fpga(remote, bitstream=None)
reconfig_time = time.time() - reconfig_start
print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))
#remote = rpc.LocalSession()

# Get execution context from remote
device = "vta"
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

#print(env.wgt_dtype, env.inp_dtype)
# 2.encoder via tvm
# ReferenceL Docs - VTA Tutorials - Optimize Tensor Operators - Matrix Multiply Blocking
# https://tvm.apache.org/docs/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-topic-vta-tutorials-optimize-matrix-multiply-opt-py

import numpy as np
#from tvm import te
from tvm import relay
#from tvm import autotvm
from vta.top import graph_pack
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
    # 1. residual
    # tvm.relay.copy(data)
    residual = relay.copy(q)
    
    # 2. weights
    #w1 = relay.var(name_hint="w1", shape=wff1_shape, dtype=env.wgt_dtype)
    #w2 = relay.var(name_hint="w2", shape=wff2_shape, dtype=env.wgt_dtype)
    
    # 3. nn.Linear
    # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    # I'm not sure of dtype
    #print("shape of q: ", relay.shape_of(q))
    #print("shape of w1: ", relay.shape_of(w1))
	
    #ret = relay.var(name_hint="relay", shape=(30, 2048), dtype=env.wgt_dtype)
    ret = relay.nn.matmul(tensor_a=q, tensor_b=w1)
    
    # 4. relu
    # tvm.relay.nn.relu(data)
    #ret_cast = relay.cast(ret, dtype=env.wgt_dtype)
    ret2 = relay.nn.relu(data=ret)
    #ret2_cast = relay.cast(ret2, dtype=env.wgt_dtype)
    # 5. nn.Linear
    #print("shape of ret: ", relay.shape_of(ret2))
    #print("shape of w2: ", relay.shape_of(w2))
    ##ret2_cast = relay.cast(ret2, dtype=env.wgt_dtype)
    q1 = relay.nn.matmul(tensor_a=ret2, tensor_b=w2)
    
    # 6. dropout
    # tvm.relay.nn.dropout(data, rate=0.5)
    q2 = relay.nn.dropout(data=q1, rate=dropout_rate)
    
    # 7. residual-sum
    # tvm.relay.add(lhs, rhis) 
    q3 = relay.add(lhs=q2, rhs=residual) 
    # 8. layer-norm
    # tvm.relay.nn.layer_norm(data, gamma, beta, axis=- 1, epsilon=1e-05, center=True, scale=True)¶
    #gamma_coef = relay.cast(relay.const(value = 1), dtype=env.wgt_dtype)
    #beta_coef = relay.cast(relay.const(value = 0), dtype=env.wgt_dtype)
    q3_cast = relay.cast(q3, dtype="float32")
    q4 = relay.nn.layer_norm(data=q3_cast, gamma=gamma, beta=beta, axis=-1, epsilon=epsilon_rate)
    #print("done")
    return q3
#def Encoder(d_model, d_ff, n_head, dk, dv, dropout):
#encoder_output = Encoder(d_model, d_ff, n_head, dk, dv, 0.1)

q_input = relay.var(name_hint="q", shape=qk_shape, dtype=env.inp_dtype)

w1 = relay.var(name_hint="w1", shape=wff1_shape, dtype=env.wgt_dtype)
w2 = relay.var(name_hint="w2", shape=wff2_shape, dtype=env.wgt_dtype)

gamma = relay.var(name_hint="gamma", shape=(d_model,), dtype="float32")
beta = relay.var(name_hint="beta", shape=(d_model,), dtype="float32")
q_output = Encoder(q_input, w1, w2, gamma, beta)

# Measure build start time
build_start = time.time()

fn_encoder = relay.Function([q_input, w1, w2, gamma, beta], q_output)
with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
    graph, lib, params = relay.build( fn_encoder, target=tvm.target.Target(env.target if device == "vta" else env.target_vta_cpu, host=env.target_host))
#fmod_encoder = tvm.IRModule.from_expr(fn_encoder)
print(graph)
print(lib)
print(params)

build_time = time.time() - build_start
print("Encoder" + " inference graph built in {0:.2f}s!".format(build_time))

temp = utils.tempdir()
lib.export_library(temp.relpath("graphlib.tar"))

remote.upload(temp.relpath("graphlib.tar"))
lib = remote.load_module("graphlib.tar")

m = graph_executor.create(graph, lib, ctx)

# seq = np.random.randint(16, size=(len_q, d_model), dtype=env.inp_dtype)
# w11 = np.random.randint(16, size=wff1_shape, dtype=env.wgt_dtype)
# w22 = np.random.randint(16, size=wff2_shape, dtype=env.wgt_dtype)
# g = np.ones((512,), dtype="float32")
# b = np.zeros((512,), dtype="float32")
# print(seq)
# print(env.BATCH)

# m.set_input("q", seq)
# m.set_input("w1", w11)
# m.set_input("w2", w22)
# m.set_input("gamma", g)
# m.set_input("beta", b)

# # Perform inference and gather execution statistics
# # More on: :py:method:`tvm.runtime.Module.time_evaluator`
# num = 1  # number of times we run module for a single measurement
# rep = 1  # number of measurements (we derive std dev from this)
# timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)
# if env.TARGET in ["sim", "tsim"]:
#     simulator.clear_stats()
#     timer()
#     sim_stats = simulator.stats()
#     print("\nExecution statistics:")
#     for k, v in sim_stats.items():
#         # Since we execute the workload many times, we need to normalize stats
#         # Note that there is always one warm up run
#         # Therefore we divide the overall stats by (num * rep + 1)
#         print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
# import sys;sys.exit(0)
# tcost = timer()
# std = np.std(tcost.results) * 1000
# mean = tcost.mean * 1000
# print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
# print("Average per sample inference time: %.2fms" % (mean))
