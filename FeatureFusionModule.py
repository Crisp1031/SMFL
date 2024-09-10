import torch
from torch import nn
class FusionModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FusionModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(input_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs):
        weights = torch.tanh(self.attention_weights(inputs))
        attention_scores = torch.softmax(self.context_vector(weights), dim=1)
        attended_inputs = inputs * attention_scores
        return attended_inputs