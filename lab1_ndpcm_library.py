from collections import namedtuple
import numpy as np

# Import quantizer for error
import lab1_library

alpha = 0.000000005
k_v = 0.00001

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

NDPCM = namedtuple('NDAPCM', ['n', 'h_depth', 'n_bits',
                   'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated'])

def init(n, h_depth, n_bits):
    # Adding values
    data_block = NDPCM(
        n, h_depth, n_bits, np.zeros((n, h_depth)), np.zeros(
            (n, h_depth)), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    )
    data_block.phi[0] = np.array([0, 0, 0])
    return data_block


def prepare_params_for_prediction(data_bloc, k):
    # Update weights for next round (k) based on previous k-1, k-2,...
    # TODO: for first iteration INITIALIZE 'phi' and 'theta'
    if (k == 1):
        data_bloc.phi[0] = np.array([0, 0, 0])
        data_bloc.theta[0] = np.array([0, 0, 0])   
        return
    if (k == 2):
        data_bloc.phi[1] = np.array([data_bloc.y_recreated[1], 0, 0])
        data_bloc.theta[1] = data_bloc.theta[0] + alpha* data_bloc.phi[0]*data_bloc.eq[1]
        return
    # TODO: Fill 'phi' history for 'h_depth' last elements
    data_bloc.phi[k] = np.array(
        [data_bloc.y_recreated[k-1]
         , data_bloc.y_recreated[k-2]
         , data_bloc.y_recreated[k-3]])
    # data_bloc.phi[k] = np.array(
    #     [data_bloc.y_recreated[k] ## Add last recreated value (y(k-1)
    #      , data_bloc.y_recreated[k-1] ## Copy shifted from previous history (y(k-2))
    #      , data_bloc.y_recreated[k-2] ])
    print("e=", data_bloc.eq[k])
    print("eT=", data_bloc.eq[k].transpose())
    # TODO: Update weights/coefficients 'theta'
    # data_bloc.theta[k] = data_bloc.theta[k-1] + alpha * data_bloc.eq[k] * np.conj(data_bloc.phi[k])
    data_bloc.theta[k] = data_bloc.theta[k-1] + alpha * data_bloc.eq[k-1] * np.conj(data_bloc.phi[k-1])


    return


def predict(data_bloc, k):
    if (k > 0):
        data_bloc.phi[k] = data_bloc.phi[k-1]
    # TODO: calculate 'hat y(k)' based on (k-1) parameters
    # data_block.y_hat[k] = ...
        # data_bloc.y_hat[k] = np.dot(data_bloc.theta[k-1], data_bloc.phi[k-1])
        data_bloc.y_hat[k] = np.dot(data_bloc.theta[k-1].transpose(), data_bloc.phi[k-1]) - k_v*data_bloc.eq[k-1]
    if (k==1):
        # data_block.y_hat[k] = ...
        # data_bloc.y_hat[k] = np.dot(data_bloc.theta[k-1], data_bloc.phi[k-1])
        data_bloc.y_hat[k] = np.dot(data_bloc.theta[0].transpose(), data_bloc.phi[0]) - k_v*data_bloc.eq[0]
    print ( data_bloc.theta[k] @ data_bloc.phi[k])
    # print("tran",data_bloc.theta[k-1].transpose() )
    # print("normal",data_bloc.theta[k-1])
    # TODO: Return prediction - fix:
    # data_bloc.y_recreated[k-1] = data_bloc.y_hat[k-1]-data_bloc.eq[k-1]
    data_bloc.y_recreated[k] = data_bloc.y_hat[k]-data_bloc.eq[k]
    return data_bloc.y_recreated[k-1];
    # return data_bloc.phi[k][0]
    # return data_bloc.y_hat[k]-data_bloc.e[k]
    # return data_bloc.y_hat[k]

def calculate_error(data_block, k, real_y):
    data_block.e[k] = real_y - data_block.y_hat[k]
    data_block.eq[k] = lab1_library.quantize(
        data_block.e[k], data_block.n_bits)
    return data_block.eq[k]


def reconstruct(data_block, k):
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]



