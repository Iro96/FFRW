import torch

def preserve_weights(old_layer, new_layer):
    with torch.no_grad():
        min_out = min(old_layer.out_features, new_layer.out_features)
        min_in = min(old_layer.in_features, new_layer.in_features)
        new_layer.weight[:min_out, :min_in] = old_layer.weight[:min_out, :min_in]
        new_layer.bias[:min_out] = old_layer.bias[:min_out]
    return new_layer
