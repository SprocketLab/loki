import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from numpy import typing as npt

class LokiExact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: torch.FloatTensor, dists: torch.FloatTensor):
        k = dists.shape[0]
        
        # Compute S(x) = vector of negative Fréchet variances
        s = -probs @ (dists ** 2)
        #print(s.shape)
        print(s)
        return s
    
    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        return grad_output, None, None
    
def loki_exact_predict(probs, dists):
    #loki = LokiExact()
    #return loki.apply(probs, dists)
    s = -probs @ (dists ** 2)
    return s

class LokiStraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: torch.FloatTensor, dists: torch.FloatTensor):
        k = dists.shape[0]
        onehot = torch.eye(k).cuda()
        
        # Compute S(x) = vector of negative Fréchet variances
        s = -probs @ (dists ** 2)

        # Intermediate decoding step
        with torch.no_grad():
            z_hat_int = s.argmax(dim=1)
            z_hat_onehot = onehot[z_hat_int]

        # Important to ensure that z_hat_smoothed gets gradients from the loss
        z_hat_onehot.requires_grad = True
        return z_hat_onehot
    
    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        return grad_output, None, None
    
def loki_ste_predict(probs, dists):
    loki = LokiStraightThroughEstimator()
    return loki.apply(probs, dists)

class LokiNegIdenEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: torch.FloatTensor, dists: torch.FloatTensor):
        k = dists.shape[0]
        onehot = torch.eye(k).cuda()
        
        # Compute S(x) = vector of negative Fréchet variances
        s = -probs @ (dists ** 2)

        # Intermediate decoding step
        with torch.no_grad():
            z_hat_int = s.argmax(dim=1)
            z_hat_onehot = onehot[z_hat_int]

        # Important to ensure that z_hat_smoothed gets gradients from the loss
        z_hat_onehot.requires_grad = True
        return z_hat_onehot
    
    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        return -grad_output, None, None
    
def loki_negiden_predict(probs, dists):
    loki = LokiNegIdenEstimator()
    return loki.apply(probs, dists)


def proj_simplex(y):
    ''' From http://www.mcduplessis.com/index.php/2016/08/22/fast-projection-onto-a-simplex-python/
    '''
    d = len(y)
    a = torch.ones(d)
    idx = torch.argsort(y)
    evalpL = lambda k: torch.sum((y[idx[k:]] - y[idx[k]])) - 1

    def bisectsearch():
        idxL, idxH = 0, d-1
        L = evalpL(idxL)
        H = evalpL(idxH)
        if L < 0:
            return idxL
        while (idxH - idxL) > 1:
            iMid = int((idxL + idxH) / 2)
            M = evalpL(iMid)
            if M > 0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M
        return idxH
        
    k = bisectsearch()
    lam = (torch.sum(y[idx[k:]]) - 1) / (d - k)
    x = F.relu(y - lam)
    return x

class LokiPolytopeEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: torch.FloatTensor,
                dists: torch.FloatTensor, eta: float):
        k = dists.shape[0]
        eta = torch.tensor(eta).cuda()
        onehot = torch.eye(k).cuda()
        
        # Compute S(x) = vector of negative Fréchet variances
        s = -probs @ (dists ** 2)

        # Intermediate decoding step
        with torch.no_grad():
            z_hat_int = s.argmax(dim=1)
            z_hat_onehot = onehot[z_hat_int]
        ctx.save_for_backward(z_hat_onehot, eta)

        # Important to ensure that z_hat_smoothed gets gradients from the loss
        z_hat_onehot.requires_grad = True
        return z_hat_onehot
    
    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        z_hat_onehot, eta, = ctx.saved_tensors
        z_unproj = z_hat_onehot - (eta * grad_output)

        # TODO project z_unproj onto the unit simplex
        # NOTE that the unit simplex is just the relaxed convex polytope of
        # the one-hot vectors. 
        # NOTE SOFTMAX IS A QUICK-AND-DIRTY NON-OPTIMAL PROJECTION...
        z_proj = z_unproj.softmax(dim=1)
        # NOTE this is the actual projection but it's slower because it's
        # not batched! 
        #b = z_unproj.shape[0]
        #z_proj = torch.stack([proj_simplex(z_unproj[i]) for i in range(b)])
        
        delta_s = z_hat_onehot - z_proj
        return delta_s, None, None, None
    
def loki_polytope_predict(probs, dists, eta=0.01):
    loki = LokiPolytopeEstimator()
    return loki.apply(probs, dists, eta)
        