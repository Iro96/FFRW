import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EvoStudent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64]):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim if i==0 else hidden_layers[i-1], h) for i,h in enumerate(hidden_layers)])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        self.activations = [F.relu for _ in hidden_layers]

    def forward(self, x):
        for layer, act in zip(self.hidden_layers, self.activations):
            x = act(layer(x))
        return self.output_layer(x)

    def mutate_add_layer(self, new_size=64):
        """Add a new layer while preserving function"""
        last_size = self.hidden_layers[-1].out_features
        new_layer = nn.Linear(last_size, new_size)
        self.hidden_layers.append(new_layer)
        self.activations.append(F.relu)
        self.output_layer = nn.Linear(new_size, self.output_layer.out_features)
