def fitness_score(model, loss, param_weight=0.01):
    params = sum(p.numel() for p in model.parameters())
    return -loss.item() + param_weight * (1/params)
