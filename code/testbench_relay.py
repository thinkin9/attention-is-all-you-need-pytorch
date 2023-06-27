# Test Benchmark for encoder_relay
# Reference
# testbench:    https://github.com/apache/tvm/blob/main/vta/tests/python/integration/test_benchmark_topi_conv2d.py
# Relay:        https://github.com/apache/tvm/tree/main/tests/python/relay

# 1. Package Load
# 2. Env load
# 3. run_encoder_relay
# 4. test_encoder_relay

# Package
import os
import pytest

import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize

# Encoder is implemented by relay
# from tvm import topi
# import tvm.topi.testing
import tvm.relay.testing 

import vta
from vta import program_fpga, reconfig_runtime

import vta.testing
from vta.testing import simulator

# 2. Env load
env = vta.get_env()

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

# # 3. run_encoder_relay
# def run_encoder_relay():
#     q = relay.var("q")

# # 4. test_encoder_relay
# def test_encoder_relay():


