

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
    # n_bits = 15;
    sim_frac = 02.051; # 1.0==100% simulation time run
    n = int(100*sim_frac);#100; # number of iterations
    h_depth = 3; ## for now hardcode size of history to last 3 values
    # max_y = 64356; ## maximum range of y=<0-MAX>; e.g.:
        ## for 16-bit ADC -> MAX=2^16 = 65356
        ## for 8-bit ADC -> MAX=2^8 = 256
        ## use max(f) to set dynamically
    # k_v = 0.001
    # alpha = 0.000001   ## a=0.001 will cause predictor/estimator to FAIL with f=<0,200> ("blow up to infinity and beyond...")

    #####################################################################
    ### Generate sample ADC data -> f = [ADC(k=0), ADC(k=1), ADC(k=2), .... ]
    x = np.linspace(0, 3*(sim_frac*pi), n)
    # useful to evaluate function at lots of points 
    f_original = np.sin(x)
    ## Scale to range 0-4095
    # f = (f_original+1)*4095
    ## Scale to range 0-100
    # f = (f_original+1)*100
    ## Scale to range 0-100
    f = (f_original+1)*100 ; ## f ranges from 0 to 200 - sinusoid

    #####################################################################
    # Check if the k_v and alpha are withing the limits 
    max_y = max(f);
    max_alpha = 1/ (h_depth * max_y*max_y);

    nu = 1/ (1- alpha * h_depth* max_y * max_y);
    max_k_v = np.sqrt(1- alpha * h_depth* max_y * max_y);
    # print("Max_y=", max_y, "  Max_alpha=", max_alpha, "  Max_k_v=", max_k_v, "  nu=", nu)

    # if (alpha > max_alpha):
    #     print("\n\nPROBLEM: Alpha is too large (a=", alpha, " > max_alpha=", max_alpha, ")\n\n")
    # if (k_v > max_k_v):
    #     print("\n\nPROBLEM: k_v is too large (k_v=", k_v, " > max_k_v=", max_k_v,")\n\n")

    #####################################################################
    ## Initialize TX and RX NADPCM structures
    phi_init = np.array([f[0], f[0], f[0]])
    tx_data = lab1_ndpcm_library.init(
        n, h_depth, n_bits, k_v, alpha, init_history=phi_init)
    rx_data = lab1_ndpcm_library.init(
        n, h_depth, n_bits, k_v, alpha, init_history=phi_init)

    #######################################################
    ## Initialize both TX/RX with first ADC sample value (has to transmit the FIRST FULL ADC sample)
    tx_data.phi[0] = np.array([f[0], f[0], f[0]])
    rx_data.phi[0] = np.array([f[0], f[0], f[0]])
    ## Clean up initial values of the variables (assuming f[0] sample is sent in full)
    tx_data.y_hat[0] = f[0]
    rx_data.y_hat[0] = f[0]
    tx_data.y_recreated[0] = f[0]
    rx_data.y_recreated[0] = f[0]
    tx_data.e[0] = 0
    rx_data.e[0] = 0
    tx_data.eq[0] = 0
    rx_data.eq[0] = 0


    e = np.zeros((n)) # Prepare array for saving true error (f - y_hat)

    # print ("Range = ", range(1, n-1))
    #####################################################################
    ## Run main simulation for iterations k=1, 2, ...., n-1
    for k in range (1, n):
        # print ("Iteration k=", k)
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


        if ((y_hat > 1.1e+50) | (y_hat < -1.1e+50)):
            print ("\n\nERROR: y_hat is 'blowing up' AT k=",k,"!!\n\n");
            e[k] = 1e255;
            break;


    ## Verify that k_v and alpha are acceptable (meet constrainsts of stability)
    print("Max_y=", max_y, "  Max_alpha=", max_alpha,
        "  Max_k_v=", max_k_v, "  nu=", nu)

    if (alpha > max_alpha):
        print("\n\nPROBLEM: Alpha is too large (a=",
            alpha, " > max_alpha=", max_alpha, ")\n\n")
    if (k_v > max_k_v):
        print("\n\nPROBLEM: k_v is too large (k_v=",
            k_v, " > max_k_v=", max_k_v, ")\n\n")


    ## For debugging print all and run for ONLY few iterations!!
    if (print_debug == 1):
        print("TX=", tx_data)
        print("RX=", rx_data)
        print("f=", f)

    #####################################################################
    ## Report some numerical metrics: throughput + errors
    e_all = np.abs(rx_data.y_recreated - f)
    e_average = np.average(e_all)
    print ("cumulative error (reconstructed y - ADC data) ="
        , e_all.sum(), " average error = ", np.average(e_all)
        );
    print ("Bits transmitted = ", n_bits*n)
    print ("Assuming 1kSamples/second bitrate [bits/sec]=", n_bits*n/1000);

    if (1 == enable_plot):
        #####################################################################
        ## Plot the results - with formatting
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
        ## error of the final reconstruction
        plt.plot(tt,rx_data.y_recreated-f, label="reconstruction error")
        plt.legend();

        plt.subplot(3, 1, 3)
        # error of the final reconstruction
        plt.plot(tt, rx_data.eq, label="quantized error")
        plt.plot(tt, e, label="true error")
        plt.legend()
        plt.show()

        plt.subplot(3, 1, 1)
        plt.title("N_bits=" + str(n_bits))
        plt.xlabel("Iteration")
        plt.ylabel("\Theta coeff")
        plt.plot(tt, tx_data.theta[:, 0], label="theta_0")
        plt.plot(tt, tx_data.theta[:, 1], label="theta_1")
        plt.plot(tt, tx_data.theta[:, 2], label="theta_2")
        plt.show()
    return e_average, rx_data
        
    tt = np.arange(1,n+1)
    return n,tt,f,rx_data
    


if __name__ == '__main__':
    a = np.array([0.000008, 0.000001])
    n_bits_range = range(2,16)
    # e_bits = np.zeros((n_bits_range.stop+1, np.shape(a)[0]))
    # for i in n_bits_range:
    #     # e_bits[i] = run_sim_ndpcm(n_bits=i, enable_plot=0)
    #     for a_i in range(np.shape(a)[0]):
    #         e_bits[i, a_i] = run_sim_ndpcm(n_bits=i, alpha=a[a_i], enable_plot=0)
    # plt.title("Average error versus n_bits")
    # plt.xlabel("n_bits")
    # plt.ylabel("Avg. error")

    # for a_i in range(np.shape(a)[0]):
    #     plt.plot(n_bits_range, e_bits[n_bits_range,
    #              a_i], label="ndpcm a="+str(a[a_i]), marker='o')
    # plt.legend()
    # plt.show()
    # run_sim_ndpcm(n_bits=15, alpha=0.00001, enable_plot=1)




    ## 计算alpha固定，比较不同nbit下重建误差
    # for i in [5]:
    #     n,tt,f,rx_data=run_sim_ndpcm(n_bits=i, alpha=0.00001, enable_plot=0)
    #     label = str(i)+" bits"
    #     plt.plot(tt,rx_data.y_recreated-f,label=label)
        
    #     plt.legend();
    # plt.ylabel("reconstruction error")
    # plt.show();

    ## 由图可知得出alpha=max_alpha时,average error 最小 此时遍历k_v 得出最合适的k_v量
    # alpha=8.33e-06
    alpha=0.000001
    h_depth=3
    max_y=200
    max_k_v = np.sqrt(1- alpha * h_depth* max_y * max_y);
    k_v_init=0.001
    average_error=[]

    # print(max_k_v)
    for i in n_bits_range:
        average_error=[]
        for k in np.arange(k_v_init, max_k_v, 0.0001):
            # print(k)
            e_average,rx_data=run_sim_ndpcm(n_bits=i, alpha=alpha, k_v=k, enable_plot=0)
            average_error.append(e_average)
        label= str(i) + " bits"
        plt.plot(np.arange(k_v_init, max_k_v, 0.0001),average_error,label=label)
        plt.legend();
    plt.title("Average error versus n_bits")
    plt.xlabel("kv")
    plt.ylabel("Avg. error")

    plt.show()