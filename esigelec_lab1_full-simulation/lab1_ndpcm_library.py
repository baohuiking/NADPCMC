
from collections import namedtuple
import numpy as np

# Import quantizer for error
import lab1_library

# Declaring namedtuple()
# n - total length of the simulation (number of samples/iterations)
# h_depth - number of history elements in \phi and corresponding coefficients (length of vectors)
# n_bits - number of bits to be transmitted (resolution of encoded error value)
# phi - vector of vectors of samples history (reproduced!!) 
#       - first index = iteration; second index = current time vector element
# theta - vector of vectors of coefficients 
#       - first index = iteration; second index = current time vector element
# y_hat - vector of all predicted (from = theta * phi + k_v * eq)
# e - exact error between the sample and the predicted value (y_hat)
# eq - quantized value of error (see n_bits!!)
# y_recreated - vector of all recreated/regenerated samples (used in the prediction!!)
##
# k_v   constant used for k_v gain
# alpha constant used for alpha gain
NDPCM = namedtuple('NDAPCM', ['n', 'h_depth', 'n_bits',
                   'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated', 'k_v', 'alpha'])


def init(n, h_depth, n_bits, a_k_v, a_alpha, init_history):
    # Adding values
    data_block = NDPCM(
        n, h_depth, n_bits, np.zeros((n, h_depth)), np.zeros(
            (n, h_depth)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), 
            k_v=a_k_v, 
            alpha=a_alpha
    )
    # data_block.phi[0] = np.array([0, 0, 0])
    # data_block.phi[0] = np.array(np.zeros(h_depth))
    data_block.phi[0] = init_history
    data_block.theta[0] = np.array( [0.4, 0.35, 0.35])
    # Modify initial value for any component, parameter:
    # data_block.k_v = k_v
    # data_block.alpha = alpha
    # ...
    return data_block


def prepare_params_for_prediction(data_block, k):
    # Update history (\phi) for this iteration (k) based on previous k-1, k-2,...
    # TODO: for first iteration INITIALIZE 'phi' and 'theta'
    if (k == 1):
         return
    # TODO: Fill 'phi' history for 'h_depth' last elements
    data_block.phi[k-1] = np.array(
        [data_block.y_recreated[k-1] ## Add last recreated value (y(k-1)
         , data_block.phi[k-2][0]  # Copy shifted from previous history (y(k-2))
         , data_block.phi[k-2][1]])
    # print("k=", k, ",Phi(k-1)=", data_block.phi[k-1])
    return


def update_weights_theta(data_block, k):
    # TODO: for first iteration INITIALIZE 'theta'
    # if (k == 1):
    #     data_block.theta[0] = np.array([0.4, 0.35, 0.35])
    #     print("Initial preparations k=1; k==", k)
    #     return
    # TODO: Update weights/coefficients 'theta'
    alpha = data_block.alpha
    data_block.theta[k] = data_block.theta[k-1] + \
        alpha*data_block.phi[k-1]*data_block.eq[k]
    # print("k=", k, ",Theta(k-1)=", data_block.theta[k])
    return

def predict(data_block, k):
    # TODO: calculate 'hat y(k)' based on (k-1) parameters
    k_v = data_block.k_v;
    data_block.y_hat[k] = data_block.theta[k-1] @ data_block.phi[k-1]+k_v * data_block.eq[k-1]
    # if (k==1):
        # data_block.y_hat[k] = ...
    # print("y_hat = ", data_block.y_hat[k])
    # print ( data_bloc.theta[k] @ data_bloc.phi[k])
    # TODO: Return prediction - fix:
    return data_block.y_hat[k]
    # return data_bloc.y_recreated[k-1];
    # return data_block.phi[k][0]


def calculate_error(data_block, k, real_y):
    data_block.e[k] = real_y - data_block.y_hat[k]
    data_block.eq[k] = lab1_library.quantize(
        data_block.e[k], data_block.n_bits)
    return data_block.eq[k]


def reconstruct(data_block, k):
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]
    return data_block.y_recreated[k]
