# THIS CODE CONTAINS:
# - Separate train and test dataset creation for the phase-noised system [implementation completed on 2021-Jun-9]
# - CNN structure that is adaptive to the MIMO size []
# - Loss function of the phase noised system fully parallel [implementation completed on 2021-Jun-7]
# - Training loop [implementation completed on 2021-Jun-8]


# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np
# tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt
# tf.distribute.Strategy

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_model import CNN_model_class
from ML_model import ML_model_class
from Sohrabi_s_method_tester import Sohrabi_s_method_tester_class
from dataset_generator import dataset_generator_class
from loss_parallel_phase_noise_free import loss_parallel_phase_noise_free_class
from loss_parallel_phase_noised import parallel_loss_phase_noised_class

# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':

    print('tf version', tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 8
    test_dataset_size = 8
    eval_dataset_size = 8
    width_of_network = 1
    BATCHSIZE = 8
    L_rate = 1e-3
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
    Nsymb = 25  # min is 2 and max is inf

    fc = 22.0e9
    c = 9.4e-19  # 4.7e-18  #
    # PHN_innovation_std = .098 # 2 * np.pi * fc * np.sqrt(c * Ts)

    f_0 = 100e3
    # # L = 10. * np.log10( (PHN_innovation_std**2)/(4.0*np.pi**2*f_0**2 ) )
    L = -85.
    fs = 32.72e6
    Ts = 1. / fs
    # tensorboard_log_frequency = 10

    PHN_innovation_std = np.sqrt(4.0 * np.pi ** 2 * f_0 ** 2 * 10 ** (L / 10.) * Ts)
    print('PHN_innovation_std = ', PHN_innovation_std)
    #
    # dataset_name = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_training_ML.mat'
    # dataset_for_testing_sohrabi = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_testing_Sohrabi.mat'

    dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'
                                   
    print('3: Testing Sohrabi\'s method started.')
    obj_Sohrabi_s_method_tester = Sohrabi_s_method_tester_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                   SNR, P, N_c, N_scatterers,
                                                   angular_spread_rad, wavelength,
                                                   d, BATCHSIZE, phase_shift_stddiv,
                                                   1, Nsymb, Ts, fc, c,
                                                   PHN_innovation_std, dataset_for_testing_sohrabi, eval_dataset_size)
    end_time_3 = time.time()
    C, C_samples_x_OFDM_index, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples = \
        obj_Sohrabi_s_method_tester.capacity_in_presence_of_phase_noise_Sohrabi()
    end_time_4 = time.time()

    print('average capacity, Sohrabi\'s method= ', C)
    print('capacity, Sohrabi\'s method= ', C_samples_x_OFDM_index)
    C_samples_x_OFDM_index = C_samples_x_OFDM_index.numpy()
    RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
    RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

    mdic = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
            "L": L,
            'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples ,
            'RQ_forall_k_forall_OFDMs_forall_samples' : RQ_forall_k_forall_OFDMs_forall_samples}

    sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/Matlab_projects/soh.mat", mdic)

    print("elapsed time of testing Sohrabi\' method = ", (end_time_4 - end_time_3), ' seconds')
    print("loss evaluation per sample = ", (end_time_4 - end_time_3)/test_dataset_size, ' seconds')
