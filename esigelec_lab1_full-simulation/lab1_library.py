
import numpy as np 
### initialize
from collections import namedtuple

# n_bits = 4;
# max_q = (2 ** (n_bits -1)) -1;
# min_q = -2 ** (n_bits -1);

# print(n_bits, min_q, max_q)

def quantize(a, n_bits):
    ## quantize a into integer with n_bits
    max_q = (2 ** (n_bits - 1)) - 1
    min_q = -2 ** (n_bits - 1)
    eq_val = round(a)
    if (a > max_q):
        eq_val = max_q
    if (a < min_q):
        eq_val = min_q
    return eq_val

