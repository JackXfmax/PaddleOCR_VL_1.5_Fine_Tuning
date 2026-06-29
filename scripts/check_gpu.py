#!/usr/bin/env python3
import paddle
print("Version:", paddle.__version__)
print("CUDA compiled:", paddle.is_compiled_with_cuda())
try:
    paddle.set_device("gpu:0")
    print("Device:", paddle.get_device())
except Exception as e:
    print("GPU error:", e)
