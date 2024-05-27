import numpy as np 
from matplotlib import pyplot as plt 
import lab1_ndpcm_library

def run_sim_ndpcm(n_bits, alpha, k_v, enable_plot):
    sim_frac = 2.051
    n = int(100 * sim_frac)
    h_depth = 3
    max_y = 64356
    
    x = np.linspace(0, 3 * (sim_frac * np.pi), n)
    f_original = np.sin(x)
    f = (f_original + 1) * 100
    
    max_y = max(f)
    max_alpha = 1 / (h_depth * max_y * max_y)
    nu = 1 / (1 - alpha * h_depth * max_y * max_y)
    max_k_v = 1 / np.sqrt(nu)
    
    phi_init = np.array([f[0], f[0], f[0]])
    tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits, k_v, alpha, init_history=phi_init)
    rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits, k_v, alpha, init_history=phi_init)
    
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
    
    e = np.zeros((n))
    
    for k in range(1, n):
        lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
        y_hat = lab1_ndpcm_library.predict(tx_data, k)
        eq = lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
        y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)
        lab1_ndpcm_library.update_weights_theta(tx_data, k)
        e[k] = f[k] - y_rec
        rx_data.eq[k] = tx_data.eq[k]
        
        lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
        y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
        y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)
        lab1_ndpcm_library.update_weights_theta(rx_data, k)
    
    e_all = np.abs(rx_data.y_recreated - f)
    e_average = np.average(e_all)
    
    if enable_plot:
        plt.plot(k_v, e_average, label=f'n_bits={n_bits}')
        plt.xlabel('K_v')
        plt.ylabel('Average Error')
        plt.title(f'Average Error vs K_v (Alpha={alpha})')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return e_average

if __name__ == '__main__':
    n_bits_values = [6, 8, 10, 12]
    alpha = 0.00001
    k_v_range = np.linspace(0.001, 0.9, 50)
    for n_bits in n_bits_values:
        avg_errors = []
        for k_v in k_v_range:
            avg_error = run_sim_ndpcm(n_bits=n_bits, alpha=alpha, k_v=k_v, enable_plot=False)
            avg_errors.append(avg_error)
        plt.plot(k_v_range, avg_errors, label=f'n_bits={n_bits}')
    
    plt.xlabel('K_v')
    plt.ylabel('Average Error')
    plt.title('Average Error vs K_v for Different n_bits')
    plt.legend()
    plt.grid(True)
    plt.show()
