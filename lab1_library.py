
import numpy as np 
### initialize
from collections import namedtuple

# n_bits = 4;
# max_q = (2 ** (n_bits -1)) -1;
# min_q = -2 ** (n_bits -1);

# print(n_bits, min_q, max_q)

# def quantize(a, n_bits):
#     ## quantize a into integer with n_bits

#     return 0

# import numpy as np 

def quantize(a, n_bits):
    # Calculate the maximum and minimum quantization levels
    max_q = (2 ** (n_bits - 1)) - 1
    min_q = -2 ** (n_bits - 1)

    # Scale the input to the quantization range
    a_scaled = np.clip(a, min_q, max_q)

    # Quantize the scaled input
    a_quantized = np.round(a_scaled).astype(int)

    return a_quantized
