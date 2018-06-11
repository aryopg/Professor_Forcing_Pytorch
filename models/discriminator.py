import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from params import *

from .additional_layers import Flatten

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, input_length):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.input_length = input_length

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.flatten = Flatten()
        self.out = nn.Linear(input_length*hidden_size, 1)

    def forward(self, x):
        outputs = torch.zeros(self.input_length, BATCH_SIZE, self.hidden_size, device=device)

        hidden = self.initHidden()
        for ei in range(self.input_length):
            embedded = self.embedding(x[ei]).view(1, BATCH_SIZE, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            outputs[ei] = output[0, 0]

        outputs = outputs.permute(1,0,2)
        feat = self.flatten(outputs)
        out = F.sigmoid(self.out(feat))

        return feat, out

    def initHidden(self):
        return torch.zeros(1, BATCH_SIZE, self.hidden_size, device=device)
