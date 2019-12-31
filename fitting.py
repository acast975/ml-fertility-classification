import pandas as pd

def undersample(X1: pd.DataFrame, X2: pd.DataFrame):
    import random
    G, L = [], []
    size = 0
    if X1.shape[0] < X2.shape[0]:
        G = X2.copy()
        L = X1.copy()
        size = X1.shape[0]
    else:
        G = X1.copy()
        L = X2.copy()
        size = X2.shape[0]
    G = G.sample(n=size, axis=0)
    return G.append(L)

def oversample(X1: pd.DataFrame, X2: pd.DataFrame, oversample_index=None):
    import random
    G, L = [], []
    size = 0
    if X1.shape[0] < X2.shape[0]:
        G = X2.copy()
        L = X1.copy()
        size = X2.shape[0]
    else:
        G = X1.copy()
        L = X2.copy()
        size = X1.shape[0]
    r_size = size
    if oversample_index is not None and G.shape[0] > L.shape[0]*oversample_index:
        r_size = L.shape[0]*oversample_index
    
    r_size = r_size - L.shape[0]
    G = G.append(L)
    L2 = L.sample(n=r_size, replace=True)
    G = G.append(L2)
    return G
