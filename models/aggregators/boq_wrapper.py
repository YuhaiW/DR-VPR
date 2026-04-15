"""
Thin wrapper around BoQ (baselines/Bag-of-Queries) so the dual-branch
pipeline can treat it like any other aggregator:
  - forward returns just the descriptor (drops attentions)
  - exposes .out_dim for generic dim queries
"""
import os
import sys
import torch.nn as nn

_BOQ_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'baselines', 'Bag-of-Queries', 'src'
)
_BOQ_SRC = os.path.abspath(_BOQ_SRC)
if _BOQ_SRC not in sys.path:
    sys.path.insert(0, _BOQ_SRC)

from boq import BoQ  # noqa: E402


class BoQWrapper(nn.Module):
    def __init__(self, in_channels=1024, proj_channels=512,
                 num_queries=64, num_layers=2, row_dim=32):
        super().__init__()
        self.boq = BoQ(
            in_channels=in_channels,
            proj_channels=proj_channels,
            num_queries=num_queries,
            num_layers=num_layers,
            row_dim=row_dim,
        )
        self.out_dim = proj_channels * row_dim

    def forward(self, x):
        desc, _ = self.boq(x)
        return desc
