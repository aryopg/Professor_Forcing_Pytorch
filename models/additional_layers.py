import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.contiguous().view(input.shape[0], -1)
