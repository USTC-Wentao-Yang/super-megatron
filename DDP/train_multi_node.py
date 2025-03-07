import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch 
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datatime

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyMode