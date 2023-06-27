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
pynq_host = os.environ.get("VTA_RPC_HOST", "192.168.1.152")
pynq_port = int(os.environ.get("VTA_RPC_PORT", "9091"))

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")
remote = rpc.connect(pynq_host, pynq_port)

# Reconfigure the JIT runtime
vta.reconfig_runtime(remote)

# Path of bitstream (Now, bitstream=None)
vta.program_fpga(remote, bitstream=None)