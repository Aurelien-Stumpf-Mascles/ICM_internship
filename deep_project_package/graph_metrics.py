import numpy as np

def redundancy(model):
    li_corr_layers = []
    for layer in model.named_parameters():
        weights = layer[1].data.cpu().numpy()
        if 'weight' in layer[0] and weights.shape[0] > 1:
            li_corr_layers.append(np.corrcoef(weights))
    return li_corr_layers