"""Stage 3: train. Usage: python scripts/train.py CONFIG [key.sub=value ...]
Multi-GPU: torchrun --nproc_per_node=N scripts/train.py CONFIG"""

import sys

from fcctrackfit.train.train import load_config, train

if len(sys.argv) < 2:
    sys.exit(__doc__)
train(load_config(sys.argv[1], sys.argv[2:]))
