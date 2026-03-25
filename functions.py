import numpy as np 

def aggregate_network_scores(net_scores):
    # net_scores = list of scores in last N seconds
    return np.mean(net_scores)


def normalize(score, min_val, max_val):
    return (score - min_val) / (max_val - min_val + 1e-8)