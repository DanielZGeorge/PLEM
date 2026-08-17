"""
losses/ — differentiable PyTorch training losses for PLEM's joint 0D/1D/2D
(point/linear/polygonal) extraction task.

Pure torch, tensor-in/tensor-out: (B, C, H, W) logits, (B, H, W) int64
targets — NOT metrics/'s H×W numpy label-map convention (see metrics/
for the evaluation-side API, which stays pure numpy/scipy/skimage and
non-differentiable by design). Each class here is a differentiable analog of
one metrics/ eval function; see each module's docstring for the specific
correspondence and what had to change to make it differentiable.
"""

from losses.tolerance import ToleranceBandLoss
from losses.boundary import SoftBoundaryLoss
from losses.cldice import SoftClDiceLoss
from losses.heatmap import PointHeatmapLoss
from losses.multitask import PLEMMultiTaskLoss

__all__ = [
    "ToleranceBandLoss",
    "SoftBoundaryLoss",
    "SoftClDiceLoss",
    "PointHeatmapLoss",
    "PLEMMultiTaskLoss",
]
