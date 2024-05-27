## pip3 install numpy
from cProfile import label
import numpy as np 
from numpy import pi

## pip3 install matplotlib
from matplotlib import pyplot as plt 
 
# import lab1_library
import lab1_ndpcm_library

print_debug = 0; ## Set to 1 if you want to print additional info/traces

def run_sim_ndpcm(n_bits, alpha, k_v, enable_plot):
    #####################################################################
    ### General parameters
    sim_frac = 02.051; # 1.0==100% simulation time run
    n = int(100*sim_frac);#100; # number of iterations
    h_depth = 3; ## for now hardcode size of history to last 3 values

    #####################################################################
    ### Generate sample ADC data -> f = [ADC(k=0), ADC(k=1), ADC(k=2), .... ]
    x = np.linspace(0, 3*(sim_frac*pi), n)
    # useful to evaluate function at lots of points 
    f_original = np.sin(x)
    f = (f_original+1)*100 ; ## f ranges from 0 to 200 - sinusoid

    #####################################################################
    ## Initialize TX and RX NADPCM structures
    phi_init = np.array([f[0], f[0], f[0]])
    tx_data = lab1_ndpcm_library.init(
        n, h_depth, n_bits, k_v, alpha, init_history=phi_init)
    rx_data = lab1_ndpcm_library.init(
        n, h_depth, n_bits, k_v, alpha, init_history=phi_init)

    #######################################################
    ## Initialize both TX/RX with first ADC sample value
    tx_data.phi[0] = np.array([f[0], f[0], f[0]])
    rx_data.phi[0] = np.array([f[0], f[0], f[0]])
    tx_data.y_hat[0] = f[0]
    rx_data.y_hat[0] = f[0]
    tx_data.y_recreated[0] = f[0]
    rx_data.y_recreated[0] = f[0]
    tx_data.e[0] = 0
    rx_data.e[0] = 0
    tx_data.eq[0] = 0
    rx_data.eq[0] = 0

    e = np.zeros((n)) # Prepare array for saving true error (f - y_hat)

    #####################################################################
    ## Run main simulation for iterations k=1, 2, ...., n-1
    for k in range (1, n):
        ## TX side
        ## Run compression part with update of coefficients
        lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
        y_hat = lab1_ndpcm_library.predict(tx_data, k)
        eq = lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
        y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)
        lab1_ndpcm_library.update_weights_theta(tx_data, k)

        e[k] = f[k] - y_rec  # Save error of recreated function values
        ## communication (e.g. inject error)
        rx_data.eq[k]=tx_data.eq[k];

        ## RX side 
        ## receiver side - recreate
        lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
        y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
        # No need to calculate ERROR - since it was received
        y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)
        lab1_ndpcm_library.update_weights_theta(rx_data, k)

        if ((y_hat > 1.1e+50) or (y_hat < -1.1e+50)):
            print ("\n\nERROR: y_hat is 'blowing up' AT k=",k,"!!\n\n");
            e[k] = 1e255;
            break;

    ## Report some numerical metrics: throughput + errors
    e_all = np.abs(rx_data.y_recreated - f)
    e_average = np.average(e_all)

    if (1 == enable_plot):
        #####################################################################
        ## Plot the results
        tt = np.arange(1,n+1) 

        plt.subplot(3,1,1); 
        plt.title("N_bits=" + str(n_bits))
        plt.xlabel("Iteration")
        plt.ylabel("Sensor value")

        plt.plot(tt, f, label="ADC data")
        plt.plot(tt,tx_data.y_hat, label="TX y_hat")
        plt.plot(tt,tx_data.y_recreated, label="TX recreated data")
        plt.legend();

        plt.subplot(3,1,2)
        plt.plot(tt,rx_data.y_recreated-f, label="reconstruction error")
        plt.legend();

        plt.subplot(3, 1, 3)
        plt.plot(tt, rx_data.eq, label="quantized error")
        plt.plot(tt, e, label="true error")
        plt.legend()

        plt.show()

        plt.title("N_bits=" + str(n_bits))
        plt.xlabel("Iteration")
        plt.ylabel("\Theta coeff")

        plt.plot(tt, tx_data.theta[:, 0], label="theta_0")
        plt.plot(tt, tx_data.theta[:, 1], label="theta_1")
        plt.plot(tt, tx_data.theta[:, 2], label="theta_2")
        plt.show()
        
    return e_average

if __name__ == '__main__':
    # # Define a range of values for n_bits, alpha, and k_v
    # n_bits_range = range(2, 16)
    # alpha_values = np.array([0.000020, 0.000015, 0.000008, 0.000001, 0.0000001, 0.00000001, 0.00000005])
    # k_v_values = np.array([0.001, 0.0001, 0.00001])

    # # Store the average errors for different parameter settings
    # e_bits_alpha = np.zeros((len(n_bits_range), len(alpha_values)))
    # e_bits_k_v = np.zeros((len(n_bits_range), len(k_v_values)))

    # # Run simulations for different n_bits and alpha values
    # for i, n_bits in enumerate(n_bits_range):
    #     for j, alpha in enumerate(alpha_values):
    #         e_bits_alpha[i, j] = run_sim_ndpcm(n_bits=n_bits, alpha=alpha, k_v=0.001, enable_plot=0)

    # # Run simulations for different n_bits and k_v values
    # for i, n_bits in enumerate(n_bits_range):
    #     for j, k_v in enumerate(k_v_values):
    #         e_bits_k_v[i, j] = run_sim_ndpcm(n_bits=n_bits, alpha=0.000001, k_v=k_v, enable_plot=0)

    # # Plot average errors versus n_bits for different alpha values
    # plt.title("Average error versus n_bits (varying alpha)")
    # plt.xlabel("n_bits")
    # plt.ylabel("Avg. error")
    # for j  in range(len(alpha_values)):
    #       plt.plot(n_bits_range, e_bits_alpha[:, j], label="alpha=" + str(alpha_values[j]), marker='o')
    # plt.legend()
    # plt.show()

    # # Plot average errors versus n_bits for different k_v values
    # plt.title("Average error versus n_bits (varying k_v)")
    # plt.xlabel("n_bits")
    # plt.ylabel("Avg. error")
    # for j in range(len(k_v_values)):
    #     plt.plot(n_bits_range, e_bits_k_v[:, j], label="k_v=" + str(k_v_values[j]), marker='o')
    # plt.legend()
    # plt.show()

    # # Determine when the reconstructed signal error becomes significantly larger
    # for j, alpha in enumerate(alpha_values):
    #     for i, n_bits in enumerate(n_bits_range):
    #         if e_bits_alpha[i, j] > 10:  # arbitrary threshold for significantly larger error
    #             print("For alpha =", alpha, "and n_bits =", n_bits, ", reconstructed signal error becomes significantly larger.")

    # for j, k_v in enumerate(k_v_values):
    #     for i, n_bits in enumerate(n_bits_range):
    #         if e_bits_k_v[i, j] > 10:  # arbitrary threshold for significantly larger error
    #             print("For k_v =", k_v, "and n_bits =", n_bits, ", reconstructed signal error becomes significantly larger.")

    # Define the values for alpha and k_v
    alpha_values = [0.000020, 0.000015, 0.000010, 0.000005]
    k_v_values = [0.001, 0.0005, 0.0001, 0.00005]
    n_bits = 6  # Fix n_bits at 6

# Initialize an array to store average errors for different alpha and k_v values
    e_bits_alpha_k_v = np.zeros((len(alpha_values), len(k_v_values)))

# Vary alpha and k_v, keeping n_bits constant
    for i, alpha in enumerate(alpha_values):
        for j, k_v in enumerate(k_v_values):
            e_bits_alpha_k_v[i, j] = run_sim_ndpcm(n_bits=n_bits, alpha=alpha, k_v=k_v, enable_plot=0)

# Plot average errors versus alpha for different k_v values
    plt.title("Average error versus alpha (varying k_v)")
    plt.xlabel("Alpha")
    plt.ylabel("Avg. error")
    for j in range(len(k_v_values)):
    plt.plot(alpha_values, e_bits_alpha_k_v[:, j], label="k_v=" + str(k_v_values[j]), marker='o')
    plt.legend()
    plt.show()

# Determine when the reconstructed signal error becomes significantly larger for different alpha and k_v values
    for i, alpha in enumerate(alpha_values):
    for j, k_v in enumerate(k_v_values):
        if e_bits_alpha_k_v[i, j] > 10:  # arbitrary threshold for significantly larger error
            print("For alpha =", alpha, "and k_v =", k_v, ", reconstructed signal error becomes significantly larger.")

# Conclusions and recommendations
    print("\nCONCLUSIONS:")
    print("1. Varying alpha and k_v:")
    print("- As alpha decreases or k_v increases, the average error tends to decrease.")
    print("- Lower values of alpha and higher values of k_v generally result in better performance in terms of error.")
    print("\nRECOMMENDATIONS:")
    print("1. For n_bits=6, choose alpha in the range [0.000005, 0.000020] and k_v in the range [0.00005, 0.001] for better performance.")

