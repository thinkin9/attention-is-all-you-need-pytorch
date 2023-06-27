## RPC Set-up 

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

env = vta.get_env()

host = os.environ.get("VTA_RPC_HOST", "192.168.1.152")
port = int(os.environ.get("VTA_RPC_PORT", "9091"))

assert tvm.runtime.enabled("rpc")
remote = rpc.connect(host, port)

vta.reconfig_runtime(remote)

# Will be replaced by path of bitstream
vta.program_fpga(remote, bitstream=None)

# set