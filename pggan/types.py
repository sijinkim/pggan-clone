from typing import Dict, Union

import torch
import torch.nn as nn


NeuralNet = nn.Module
Optimizer = torch.optim.Optimizer
StateDict = Dict[str, Union[Dict, int]]
