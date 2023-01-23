import torch

def smooth_hard_preds(preds, k, eps=0.1):
    bs = preds.shape[0]
    rest = (1 / (k - 1)) * eps
    p_rest = torch.ones((bs, k)) * rest
    onehot = torch.eye(k)[preds]
    onehot_scaled = onehot * ((1.0 - eps) - (1 / (k - 1)) * eps)
    p_smooth = p_rest + onehot_scaled
    assert((p_smooth.argmax(dim=1) == preds).all())
    # Important that p_smooth requires grad so that it can be used in SPIGOT
    p_smooth.requires_grad = True
    return p_smooth