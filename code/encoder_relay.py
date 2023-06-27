# ENCODER_RELAY

# Load VTA parameters (3rdparty/vta-hw/config/vta_config.json file)
import vta
env = vta.get_env()

# 2.encoder via tvm
# ReferenceL Docs - VTA Tutorials - Optimize Tensor Operators - Matrix Multiply Blocking
# https://tvm.apache.org/docs/topic/vta/tutorials/optimize/matrix_multiply_opt.html#sphx-glr-topic-vta-tutorials-optimize-matrix-multiply-opt-py

import numpy as np
import tvm
#from tvm import te
from tvm import relay
#from tvm import autotvm

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

qk_shape_head = (len_q, n_head, dk)
v_shape_head = (len_q, n_head, dv)  # Actually, qk_shape == v_shape

def Encoder(q_input, wq, wk, wv, fc, w1, w2):
    
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
    q = q_input
    k = relay.copy(q)
    v = relay.copy(q)
    
    # 2. residual
    # tvm.relay.copy(data)
    residual = relay.copy(q)
    
    # 3. weights
    # I'm not sure of dtype
    # tvm.relay.var(name_hint, type_annotation=None, shape=None, dtype='float32', span=None)
    # wq = relay.var(name_hint="wq", shape=w_shape, dtype=env.wgt_dtype)
    # wk = relay.var(name_hint="wk", shape=w_shape, dtype=env.wgt_dtype)
    # wv = relay.var(name_hint="wv", shape=w_shape, dtype=env.wgt_dtype)
    
    # 4. nn.Linear
    # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    # I'm not sure of dtype
    q = relay.nn.dense(data=q, weight=wq)
    k = relay.nn.dense(data=k, weight=wk)
    v = relay.nn.dense(data=v, weight=wv)
    
    # 5. View as head
    # Pass through the pre-attention projection: len_q x (n_head*dk)
    # Separate different heads: len_q x n_head x dk
    # Transpose for attention dot product: n_head x len_q x dv
    
    # tvm.relay.reshape(data, newshape, allowzero=False)
    # tvm.relay.transpose(data, axes=None)
    q_heads = relay.transpose(data=relay.reshape(data=q, newshape=qk_shape_head), axes=(1, 0, 2))
    k_heads = relay.transpose(data=relay.reshape(data=k, newshape=qk_shape_head), axes=(1, 0, 2))
    v_heads = relay.transpose(data=relay.reshape(data=v, newshape=v_shape_head), axes=(1, 0, 2))


    # 6. ScaledDotProductAttention
    
    # 6.1. matmul:    Q x K.T
    # tvm.relay.nn.matmul(tensor_a, tensor_b, units=None, out_dtype='', transpose_a=False, transpose_b=False)
    # tvm.relay.nn.batch_matmul(tensor_a, tensor_b, out_dtype='', transpose_a=False, transpose_b=True)
    attn = relay.nn.batch_matmul(tensor_a=q_heads, tensor_b=relay.transpose(data=k_heads, axes=(0, 2, 1)))
    
    # 6.2. scale:     sqrt(dk)
    # sqrt(64) = 8
    # tvm.relay.divide(lhs, rhs)
    scale_coef = relay.const(value = 1 / 8)
    attn = relay.multiply(q, scale_coef)
    #attn = relay.divide(lhs=attn, rhs=dk ** 0.5)
    
    # 6.3. mask:      Not implemented (Mask only for Decoder)
    
    # 6.4. Softmax:   relay.nn.softmax
    # tvm.relay.nn.softmax(data, axis=- 1)
    attn = relay.nn.softmax(data=attn, axis=-1)
    
    # 6.5. matmul:    attn x V
    # tvm.relay.nn.matmul(tensor_a, tensor_b, units=None, out_dtype='', transpose_a=False, transpose_b=False)
    # tvm.relay.nn.batch_matmul(tensor_a, tensor_b, out_dtype='', transpose_a=False, transpose_b=True)
    output = relay.nn.batch_matmul(tensor_a=attn, tensor_b=v_heads)
    
    # 7. concate
    # output: n_head x len_q x dv (current) -> len_q x n_head x dv -> len_q x (n_head*dk)
    # tvm.relay.transpose(data, axes=None)
    # tvm.relay.reshape(data, newshape, allowzero=False)
    q = relay.reshape(data=relay.transpose(data=output, axes=(1, 0, 2)), newshape=qk_shape)
    
    # 8. Linear
    # tvm.relay.var(name_hint, type_annotation=None, shape=None, dtype='float32', span=None)
    # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    #fc = relay.var(name_hint="fc", shape=fc_shape, dtype=env.wgt_dtype)
    q = relay.nn.dense(data=q, weight=fc)
    
    # 9. dropout
    # tvm.relay.nn.dropout(data, rate=0.5)
    q = relay.nn.dropout(data=q, rate=dropout_rate)
    
    # 10. residual-sum
    # tvm.relay.add(lhs, rhs)
    q = relay.add(lhs=q, rhs=residual)
    
    # 11. layer-norm
    # tvm.relay.nn.layer_norm(data, gamma, beta, axis=- 1, epsilon=1e-05, center=True, scale=True)¶
    gamma_coef = relay.const(value = 1)
    beta_coef = relay.const(value = 0)
    q = relay.nn.layer_norm(data=q, gamma=gamma_coef, beta=beta_coef, axis=-1, epsilon=epsilon_rate)
    
    
    # FeedForward:
    
    # 1. residual
    # tvm.relay.copy(data)
    residual = relay.copy(q)
    
    # 2. weights
    #w1 = relay.var(name_hint="w1", shape=wff1_shape, dtype=env.wgt_dtype)
    #w2 = relay.var(name_hint="w2", shape=wff2_shape, dtype=env.wgt_dtype)
    
    # 3. nn.Linear
    # tvm.relay.nn.dense(data, weight, units=None, out_dtype='')
    # I'm not sure of dtype
    q = relay.nn.dense(data=q, weight=w1)
    
    # 4. relu
    # tvm.relay.nn.relu(data)
    q = relay.nn.relu(data=q)
    
    # 5. nn.Linear
    q = relay.nn.dense(data=q, weight=w2)
    
    # 6. dropout
    # tvm.relay.nn.dropout(data, rate=0.5)
    q = relay.nn.dropout(data=q, rate=dropout_rate)
    
    # 7. residual-sum
    # tvm.relay.add(lhs, rhs)
    q = relay.add(lhs=q, rhs=residual)
    
    # 8. layer-norm
    # tvm.relay.nn.layer_norm(data, gamma, beta, axis=- 1, epsilon=1e-05, center=True, scale=True)¶
    q = relay.nn.layer_norm(data=q, gamma=gamma_coef, beta=beta_coef, axis=-1, epsilon=epsilon_rate)
    #print("done")
    return q

#def Encoder(d_model, d_ff, n_head, dk, dv, dropout):
#encoder_output = Encoder(d_model, d_ff, n_head, dk, dv, 0.1)

q_input = relay.var(name_hint="q", shape=qk_shape, dtype=env.inp_dtype)

wq = relay.var(name_hint="wq", shape=w_shape, dtype=env.wgt_dtype)
wk = relay.var(name_hint="wk", shape=w_shape, dtype=env.wgt_dtype)
wv = relay.var(name_hint="wv", shape=w_shape, dtype=env.wgt_dtype)
fc = relay.var(name_hint="fc", shape=fc_shape, dtype=env.wgt_dtype)

w1 = relay.var(name_hint="w1", shape=wff1_shape, dtype=env.wgt_dtype)
w2 = relay.var(name_hint="w2", shape=wff2_shape, dtype=env.wgt_dtype)

q_output = Encoder(q_input, wq, wk, wv, fc, w1, w2)

fn_encoder = relay.Function([q_input, wq, wk, wv, fc, w1, w2], q_output)
fmod_encoder = tvm.IRModule.from_expr(fn_encoder)

#encoder_mod = relay.Module.from_expr(encoder_output)
with tvm.transform.PassContext(opt_level=3):
    mod = relay.build(fmod_encoder, target="llvm")