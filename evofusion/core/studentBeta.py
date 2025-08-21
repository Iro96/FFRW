import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable

class EvoStudent(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_layers: List[int] = [64, 64], 
        activation: Callable = F.relu,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout

        # Create hidden layers dynamically
        self.hidden_layers = nn.ModuleList()
        in_dim = input_dim
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(in_dim, h))
            in_dim = h

        # Optional dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in hidden_layers]) if dropout > 0 else None

        # Output layer
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if self.dropout_rate > 0:
                x = self.dropouts[i](x)
        return self.output_layer(x)

    def mutate_add_layer(self, new_size: int = 64):
        """
        Add a new hidden layer at the end before the output layer.
        """
        last_size = self.hidden_layers[-1].out_features if self.hidden_layers else self.input_dim

        # Create new layer and dropout
        new_layer = nn.Linear(last_size, new_size)
        self.hidden_layers.append(new_layer)
        if self.dropout_rate > 0:
            self.dropouts.append(nn.Dropout(self.dropout_rate))

        # Adjust output layer
        self.output_layer = nn.Linear(new_size, self.output_layer.out_features)
        self.hidden_sizes.append(new_size)

        print(f"[Mutation] Added new layer: {last_size} -> {new_size}")

    def mutate_layer_size(self, layer_idx: int, new_size: int):
        """
        Mutate an existing hidden layer to a new size.
        """
        if layer_idx < 0 or layer_idx >= len(self.hidden_layers):
            raise IndexError("Layer index out of range")

        in_features = self.hidden_layers[layer_idx].in_features
        self.hidden_layers[layer_idx] = nn.Linear(in_features, new_size)
        self.hidden_sizes[layer_idx] = new_size

        # Adjust subsequent layer or output layer
        if layer_idx + 1 < len(self.hidden_layers):
            next_in_features = self.hidden_layers[layer_idx + 1].out_features
            self.hidden_layers[layer_idx + 1] = nn.Linear(new_size, next_in_features)
        else:
            self.output_layer = nn.Linear(new_size, self.output_layer.out_features)

        print(f"[Mutation] Resized layer {layer_idx} to {new_size} units")
