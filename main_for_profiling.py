# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np
from line_profiler import LineProfiler

tf.config.run_functions_eagerly(True)
# import matplotlib.pyplot as plt
# tf.distribute.Strategy

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> the device name: ',
      tf.config.list_physical_devices('GPU'))
if tf.test.gpu_device_name() == '/device:GPU:0':
    tf.device('/device:GPU:0')

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_model import CNN_model_class
from ML_model import ML_model_class
from Sohrabi_s_method_tester import Sohrabi_s_method_tester_class
from dataset_generator import dataset_generator_class
from loss_parallel_phase_noise_free import loss_parallel_phase_noise_free_class
from loss_parallel_phase_noised import parallel_loss_phase_noised_class
from loss_sequential_phase_noised import sequential_loss_phase_noised_class

# from profiling import profiling_class


# tf.debugging.set_log_device_placement(True)


# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    on_what_device = 'cpu'

    print('-- tf version', tf.version.VERSION)
    print("-- Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 8
    test_dataset_size = 8
    eval_dataset_size = 8
    width_of_network = 5
    BATCHSIZE = 4
    L_rate =  1e-3
    dropout_rate = 0.5
    precision_fixer = 1e-6
    # tensorboard_log_frequency = 1

    # PARAMETERS ///////////////////////////////////////////////////////////////////////////////////////////////////////
    N_b_a = 4
    N_b_rf = 2
    N_b_o = N_b_rf
    N_u_a = 4
    N_u_rf = 2
    N_u_o = N_u_rf
    N_s = 1
    K = 4
    SNR = 20.
    P = 100.
    sigma2 = 1.  # P / (10 ** (SNR / 10.))
    N_c = 5
    N_scatterers = 10
    angular_spread_rad = 0.1745  # 10deg
    wavelength = 1.
    d = .5
    phi_c = .01
    phase_shift_stddiv = 0.0
    Nsymb = 50  # min is 2 and max is inf

    fc = 22.0e9
    c = 9.4e-19  # 4.7e-18  #
    # PHN_innovation_std = .098 # 2 * np.pi * fc * np.sqrt(c * Ts)

    f_0 = 100e3
    # # L = 10. * np.log10( (PHN_innovation_std**2)/(4.0*np.pi**2*f_0**2 ) )
    L = -85.
    fs = 32.72e6
    Ts = 1. / fs
    # tensorboard_log_frequency = 10


    PHN_innovation_std = np.sqrt(1024/4)*np.sqrt(4.0 * np.pi ** 2 * f_0 ** 2 * 10 ** (L / 10.) * Ts)
    print('-- PHN_innovation_std = ', PHN_innovation_std)

    if (on_what_device == 'cpu'):
        dataset_name = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_training_ML.mat'
        dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'
    else:
        dataset_name = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_training_ML.mat'
        dataset_for_testing_sohrabi = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_testing_Sohrabi.mat'


    # Truncation and sampling of sums
    truncation_ratio_keep = 4 / K
    sampling_ratio_time_domain_keep = 4 / Nsymb
    sampling_ratio_subcarrier_domain_keep = 4 / K

    print('-- Parameter initialization is done.')

    obj_capacity_metric = sequential_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                             N_scatterers, angular_spread_rad, wavelength,
                                                             d, BATCHSIZE, 1, Nsymb, 1, 1, 'train')

    # The profiling starts from here /////////////////////////////////////////////////////////////////////////////////////


    # profiling capacity_forall_samples --------------------------------------------------------------
    LP_capacity_forall_samples = LineProfiler()
    LP_WRAPPER_capacity_forall_samples = LP_capacity_forall_samples(obj_capacity_metric.capacity_forall_samples)
    # preparing inputs
    V_D = tf.complex(tf.random.normal(shape=[BATCHSIZE ,K, N_b_rf, N_s], dtype=tf.float32),
                       tf.random.normal(shape=[BATCHSIZE, K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[BATCHSIZE, K, N_u_rf, N_s], dtype=tf.float32),
                       tf.random.normal(shape=[BATCHSIZE, K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[BATCHSIZE, N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[BATCHSIZE, N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[BATCHSIZE, N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[BATCHSIZE, N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[BATCHSIZE, K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[BATCHSIZE, K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[BATCHSIZE, Nsymb, K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[BATCHSIZE, Nsymb, K, N_b_a, N_b_a], dtype=tf.float32))

    Lambda_U = tf.complex(tf.random.normal(shape=[BATCHSIZE, Nsymb, K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[BATCHSIZE, Nsymb, K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
    # dummy run
    obj_capacity_metric.capacity_forall_samples(inputs)
    # profiler run
    LP_WRAPPER_capacity_forall_samples(inputs)
    LP_capacity_forall_samples.print_stats(output_unit=1e-6)




    # profiling capacity_forall_symbols --------------------------------------------------------------
    LP_capacity_forall_symbols = LineProfiler()
    LP_WRAPPER_capacity_forall_symbols = LP_capacity_forall_symbols(obj_capacity_metric.capacity_forall_symbols)
    # preparing inputs
    # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U
    V_D = tf.complex(tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[Nsymb, K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[Nsymb, K, N_b_a, N_b_a], dtype=tf.float32))
    Lambda_U = tf.complex(tf.random.normal(shape=[Nsymb, K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[Nsymb, K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
    # dummy run
    obj_capacity_metric.capacity_forall_symbols(inputs)
    # profiler run
    LP_WRAPPER_capacity_forall_symbols(inputs)
    LP_capacity_forall_symbols.print_stats(output_unit=1e-6)




    # profiling capacity_forall_k --------------------------------------------------------------
    LP_capacity_forall_k = LineProfiler()
    LP_WRAPPER_capacity_forall_k = LP_capacity_forall_k(obj_capacity_metric.capacity_forall_k)
    # preparing inputs
    # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U
    V_D = tf.complex(tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32))
    Lambda_U = tf.complex(tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
    # dummy run
    obj_capacity_metric.capacity_forall_k(inputs)
    # profiler run
    LP_WRAPPER_capacity_forall_k(inputs)
    LP_capacity_forall_k.print_stats(output_unit=1e-6)



    # profiling capacity_and_RX_RQ_per_k --------------------------------------------------------------
    LP_capacity_and_RX_RQ_per_k = LineProfiler()
    LP_WRAPPER_capacity_and_RX_RQ_per_k = LP_capacity_and_RX_RQ_per_k(obj_capacity_metric.capacity_and_RX_RQ_per_k)
    # preparing inputs
    V_D = tf.complex(tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32))
    Lambda_U = tf.complex(tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, 0]
    # dummy run
    obj_capacity_metric.capacity_and_RX_RQ_per_k(inputs)
    # profiler run
    LP_WRAPPER_capacity_and_RX_RQ_per_k(inputs)
    LP_capacity_and_RX_RQ_per_k.print_stats(output_unit=1e-6)

    # profiling Rq_per_k --------------------------------------------------------------
    LP_Rq_per_k = LineProfiler()
    LP_WRAPPER_Rq_per_k = LP_Rq_per_k(obj_capacity_metric.Rq_per_k)
    # preparing inputs
    V_D = tf.complex(tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_b_rf, N_s], dtype=tf.float32))
    W_D = tf.complex(tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[K, N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32))
    Lambda_U = tf.complex(tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, 0]
    # dummy run
    obj_capacity_metric.Rq_per_k(inputs)
    # profiler run
    LP_WRAPPER_Rq_per_k(inputs)
    LP_Rq_per_k.print_stats(output_unit=1e-6)



    # profiling R_I_Q_m_k --------------------------------------------------------------
    LP_R_I_Q_m_k = LineProfiler()
    LP_WRAPPER_R_I_Q_m_k = LP_R_I_Q_m_k(obj_capacity_metric.R_I_Q_m_k)
    # preparing inputs
    V_D_m = tf.complex(tf.random.normal(shape=[N_b_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[N_b_rf, N_s], dtype=tf.float32))
    W_D_k = tf.complex(tf.random.normal(shape=[N_u_rf, N_s], dtype=tf.float32),
                     tf.random.normal(shape=[N_u_rf, N_s], dtype=tf.float32))
    V_RF = tf.complex(tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_b_a, N_b_rf], dtype=tf.float32))
    W_RF = tf.complex(tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32),
                      tf.random.normal(shape=[N_u_a, N_u_rf], dtype=tf.float32))
    H = tf.complex(tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32),
                   tf.random.normal(shape=[K, N_u_a, N_b_a], dtype=tf.float32))
    Lambda_B = tf.complex(tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_b_a, N_b_a], dtype=tf.float32))
    Lambda_U = tf.complex(tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32),
                          tf.random.normal(shape=[K, N_u_a, N_u_a], dtype=tf.float32))

    inputs = [V_D_m, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, 0, 1]
    # dummy run
    obj_capacity_metric.R_I_Q_m_k(inputs)
    # profiler run
    LP_WRAPPER_R_I_Q_m_k(inputs)
    LP_R_I_Q_m_k.print_stats(output_unit=1e-6)


    # # profiling --------------------------------------------------------------
    # LP_ = LineProfiler()
    # LP_WRAPPER_ = LP_(obj_capacity_metric.)
    # dummy run
    # profiler run
    # LP_WRAPPER_()
    # LP_.print_stats(output_unit=1e-6)
