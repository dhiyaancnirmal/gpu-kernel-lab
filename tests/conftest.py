import os

import torch

# Without a GPU, run Triton kernels in Triton's CPU interpreter. Must be set
# before triton is first imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
