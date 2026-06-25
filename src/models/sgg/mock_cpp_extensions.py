
import torch
from torchvision.ops import nms as torch_nms

class MockCppExtensions:
    @staticmethod
    def nms(boxes, scores, thresh):
        return torch_nms(boxes, scores, thresh)

# Other functions can be added as needed
