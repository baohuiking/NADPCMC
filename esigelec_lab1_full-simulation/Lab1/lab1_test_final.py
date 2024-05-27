import numpy as np
import matplotlib.pyplot as plt
import lab1_ndpcm_library
from numpy import pi

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

import numpy as np
import matplotlib.pyplot as plt
import lab1_ndpcm_library

def run_simulations():
    n_bits_range = range(4, 8)
    alphas = np.linspace(0.00000001, 0.000020, num=100)  # alpha in the range [0.00000001, 0.000020]
    k_vs = np.linspace(0.00001, 0.001, num=100)  # k_v in the range [0.00001, 0.001]

    # Prepare a dictionary to store the results
    results = {}

    for n_bits in n_bits_range:
        for alpha in alphas:
            for k_v in k_vs:
                e_avg = run_sim_ndpcm(n_bits=n_bits, alpha=alpha, k_v=k_v, enable_plot=0)
                print(f"Average error with {n_bits} bits, ALPHA={alpha} and K V gain={k_v}: {e_avg}")

                # Store the results in the dictionary
                results[(n_bits, alpha, k_v)] = e_avg

    # Plot the results
    plt.figure(figsize=(10, 6))
    for n_bits in n_bits_range:
        # Prepare lists to store the x and y values for the plot
        x_values_alpha = []
        y_values = []

        for alpha in alphas:
            # Append the alpha values to the x_values list
            x_values_alpha.append(alpha)

            # Calculate the average error for the current alpha value and append it to the y_values list
            avg_error = np.mean([results[(n_bits, alpha, k_v)] for k_v in k_vs])
            y_values.append(avg_error)

        # Create a line plot for the current n_bits value
        plt.plot(x_values_alpha, y_values, label=f"{n_bits} bits")

    plt.title("Average error for different bit sizes")
    plt.xlabel("ALPHA")
    plt.ylabel("Average error")
    plt.legend()
    plt.show()

run_simulations()

    
