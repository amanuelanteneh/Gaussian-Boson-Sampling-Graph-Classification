import numpy as np

"""
Function that returns the number of samples needed to apporximate a probability distritbuion of D outcomes 
with probability at most delta that the 
L1 distance between the empirical distribution and the true distribution is at most epsilon (eps).
"""
def samplesNeeded(D, delta, eps):
    s = np.ceil( (2*(np.log(2)*D + np.log(1/delta))) /  eps**2 )
    return(int(s))

