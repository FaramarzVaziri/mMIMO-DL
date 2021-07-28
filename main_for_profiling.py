# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np
# import cProfile, pstats, io
from line_profiler import LineProfiler
# import line_profiler
# import tensorflow.experimental.numpy as tnp

# tf.config.run_functions_eagerly(True)
# import matplotlib.pyplot as plt
# tf.distribute.Strategy

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_model import CNN_model_class
from ML_model import ML_model_class
from Sohrabi_s_method_tester import Sohrabi_s_method_tester_class
from dataset_generator import dataset_generator_class
from loss_parallel_phase_noise_free import loss_parallel_phase_noise_free_class
from loss_parallel_phase_noised import paralle_loss_phase_noised_class
# from profiling import profiling_class
from loss_sequential_phase_noised import sequential_loss_phase_noised_class

# tf.debugging.set_log_device_placement(True)


# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    print('tf version', tf.version.VERSION)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 2  # int(input("No. train samples: "))
    test_dataset_size = 2  # int(input("No. test samples: "))
    width_of_network = 1  # float(input("Network's width parameter: "))
    BATCHSIZE = 1  # int(input("batch size: "))
    L_rate = 1e-4  # float(input("inital lr: "))
    dropout_rate = .5  # float(input("dropout rate: "))
    precision_fixer = 1e-6  # float(input("precision fixer additive: "))
    # tensorboard_log_frequency = 1
    # PARAMETERS ///////////////////////////////////////////////////////////////////////////////////////////////////////
    N_b_a = 16
    N_b_rf = 2
    N_b_o = N_b_rf
    N_u_a = 16
    N_u_rf = 2
    N_u_o = N_u_rf
    N_s = 1
    K = 64
    SNR = 20.
    P = 100.
    sigma2 = 1. #P / (10 ** (SNR / 10.))
    N_c = 5
    N_scatterers = 10
    angular_spread_rad = 0.1745  # 10deg
    wavelength = 1.
    d = .5
    phi_c = .01
    phase_shift_stddiv = 0.0
    Nsymb = 50  # min is 2 and max is inf
    fc = 22.0e9
    c = 9.4e-19 #4.7e-18  #
    # PHN_innovation_std = .098 # 2 * np.pi * fc * np.sqrt(c * Ts)
    f_0 = 100e3
    # # L = 10. * np.log10( (PHN_innovation_std**2)/(4.0*np.pi**2*f_0**2 ) )
    L = -85.
    fs = 32.72e6
    Ts = 1. / fs
    # tensorboard_log_frequency = 10
    PHN_innovation_std = np.sqrt( 4.0*np.pi**2*f_0**2 * 10**(L/10.) * Ts)
    # print('PHN_innovation_std = ', PHN_innovation_std)


    dataset_name = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_training_ML.mat'
    dataset_for_testing_sohrabi = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_testing_Sohrabi.mat'

    # dataset_name = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_training_ML.mat'
    # dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'

    # Truncation and sampling of sums
    truncation_ratio_keep = 1
    sampling_ratio_time_domain_keep = 1
    sampling_ratio_subcarrier_domain_keep = 1

    obj_dataset_test_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                   angular_spread_rad, wavelength, d, BATCHSIZE,
                                                   phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts,
                                                   fc,
                                                   c, PHN_innovation_std, dataset_name, test_dataset_size)
    _, H_tilde_0_complex, H_complex, Lambda_B, Lambda_U = obj_dataset_test_phn.dataset_generator(mode="test", phase_noise="y")
    print('STEP 5: Dataset creation is done.')


    # The profiling starts from here /////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    # Loss object
    obj_sequential_loss_phase_noised = sequential_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                                     N_scatterers, angular_spread_rad, wavelength,
                                                                     d, BATCHSIZE, phase_shift_stddiv,
                                                                     truncation_ratio_keep, Nsymb,
                                                                     sampling_ratio_time_domain_keep,
                                                                     sampling_ratio_subcarrier_domain_keep)


    # profiling cyclical_shift
    LP_cyclical_shift = LineProfiler()
    LP_WRAPPER_cyclical_shift = LP_cyclical_shift(obj_sequential_loss_phase_noised.cyclical_shift)
    # dummy_run
    obj_sequential_loss_phase_noised.cyclical_shift(Lambda_matrix=Lambda_U[0,0,:],k=1, flip=True)
    # profiler run
    LP_WRAPPER_cyclical_shift(Lambda_matrix=Lambda_U[0,0,:],k=1, flip=True)
    LP_cyclical_shift.print_stats(output_unit=1e-6)

    # profiling non_zero_element_finder_for_H_tilde
    LP_non_zero_element_finder_for_H_tilde = LineProfiler()
    LP_WRAPPER_non_zero_element_finder_for_H_tilde = LP_non_zero_element_finder_for_H_tilde(obj_sequential_loss_phase_noised.non_zero_element_finder_for_H_tilde)
    # dummy run
    obj_sequential_loss_phase_noised.non_zero_element_finder_for_H_tilde(k = 1, truncation_ratio_keep = 1)
    # profiler run
    LP_WRAPPER_non_zero_element_finder_for_H_tilde(k = 1, truncation_ratio_keep = 1)
    LP_non_zero_element_finder_for_H_tilde.print_stats(output_unit=1e-6)

    # profiling H_tilde_k_calculation
    LP_H_tilde_k_calculation = LineProfiler()
    LP_WRAPPER_H_tilde_k_calculation = LP_H_tilde_k_calculation(obj_sequential_loss_phase_noised.H_tilde_k_calculation)
    # dummy run
    obj_sequential_loss_phase_noised.H_tilde_k_calculation([H_tilde_0_complex[0,0,:], Lambda_B[0,0,:], Lambda_U[0,0,:]])
    # profiler run
    LP_WRAPPER_H_tilde_k_calculation([H_tilde_0_complex[0,0,:], Lambda_B[0,0,:], Lambda_U[0,0,:]])
    LP_H_tilde_k_calculation.print_stats(output_unit=1e-6)


    # profiling Rx_calculation_per_k
    LP_Rx_calculation_per_k = LineProfiler()
    LP_WRAPPER_Rx_calculation_per_k = LP_Rx_calculation_per_k(obj_sequential_loss_phase_noised.Rx_calculation_per_k)
    # preparing inputs
    V_D_k = tf.complex(tf.random.normal(shape=[N_b_rf, N_s], dtype=tf.float32), tf.random.normal(shape=[N_b_rf, N_s], dtype=tf.float32))
    W_D_k = tf.complex(tf.random.normal(shape=[N_u_rf, N_s], dtype=tf.float32), tf.random.normal(shape=[N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32), tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32), tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    inputs = [V_D_k, W_D_k, H_tilde_0_complex[0,:], V_RF, W_RF, Lambda_B[0,0,:], Lambda_U[0,0,:], 1]
    # dummy run
    obj_sequential_loss_phase_noised.Rx_calculation_per_k(inputs)
    # profiler run
    LP_WRAPPER_Rx_calculation_per_k(inputs)
    LP_Rx_calculation_per_k.print_stats(output_unit=1e-6)



    # profiling Rx_calculation_forall_k
    LP_Rx_calculation_forall_k = LineProfiler()
    LP_WRAPPER_Rx_calculation_forall_k = LP_Rx_calculation_forall_k(obj_sequential_loss_phase_noised.Rx_calculation_forall_k)
    # preparing inputs
    # preparing inputs
    V_D = tf.complex(tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32),
                       tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32),
                       tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    sampled_K = np.random.choice(K, int(sampling_ratio_subcarrier_domain_keep * K), replace=False)
    inputs = [ V_D, W_D, H_tilde_0_complex[0,:], V_RF, W_RF, Lambda_B[0,0,:], Lambda_U[0,0,:], sampled_K]
    # dummy run
    obj_sequential_loss_phase_noised.Rx_calculation_forall_k(inputs)
    # profiler run
    LP_WRAPPER_Rx_calculation_forall_k(inputs)
    LP_Rx_calculation_forall_k.print_stats(output_unit=1e-6)

    # profiling non_zero_element_finder_for_H_hat
    LP_non_zero_element_finder_for_H_hat = LineProfiler()
    LP_WRAPPER_non_zero_element_finder_for_H_hat = LP_non_zero_element_finder_for_H_hat(obj_sequential_loss_phase_noised.non_zero_element_finder_for_H_hat)
    # dummy run
    obj_sequential_loss_phase_noised.non_zero_element_finder_for_H_hat(k =1, m =0, truncation_ratio_keep=1)
    # profiler run
    LP_WRAPPER_non_zero_element_finder_for_H_hat(k =1, m =0, truncation_ratio_keep=1)
    LP_non_zero_element_finder_for_H_hat.print_stats(output_unit=1e-6)

    # profiling H_hat_m_k_calculation
    LP_H_hat_m_k_calculation = LineProfiler()
    LP_WRAPPER_H_hat_m_k_calculation = LP_H_hat_m_k_calculation(obj_sequential_loss_phase_noised.H_hat_m_k_calculation)
    # dummy run
    obj_sequential_loss_phase_noised.H_hat_m_k_calculation([H_tilde_0_complex[0,0,:], Lambda_B[0,0,:], Lambda_U[0,0,:]])
    # profiler run
    LP_WRAPPER_H_hat_m_k_calculation([H_tilde_0_complex[0,0,:], Lambda_B[0,0,:], Lambda_U[0,0,:]])
    LP_H_hat_m_k_calculation.print_stats(output_unit=1e-6)

    # # profiling
    # LP_ = LineProfiler()
    # LP_WRAPPER_ = LP_(obj_sequential_loss_phase_noised.)
    # dummy run
    # profiler run
    # LP_WRAPPER_()
    # LP_.print_stats(output_unit=1e-6)