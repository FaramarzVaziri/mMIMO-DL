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
from loss_parallel_phase_noised import paralle_loss_phase_noised_class

# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    print('The main code for running on this computer, Sohrabi\'s method')
    print('tf version', tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 10  # int(input("No. train samples: "))
    test_dataset_size = 256  # int(input("No. test samples: "))
    width_of_network = 1  # float(input("Network's width parameter: "))
    BATCHSIZE = 32  # int(input("batch size: "))
    L_rate = 1e-5  # float(input("inital lr: "))
    dropout_rate = .5  # float(input("dropout rate: "))
    precision_fixer = 1e-12  # float(input("precision fixer additive: "))

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
    tensorboard_log_frequency = 10

    PHN_innovation_std = np.sqrt( 4.0*np.pi**2*f_0**2 * 10**(L/10.) * Ts)
    print('PHN_innovation_std = ', PHN_innovation_std)

    # dataset_name = '/project/st-lampe-1/Faramarz/data/dataset/DS_for_py_for_training_ML.mat'
    # dataset_for_testing_sohrabi = '/project/st-lampe-1/Faramarz/data/dataset/DS_for_py_for_testing_Sohrabi.mat'

    dataset_name = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_training_ML.mat'
    dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'

    # Truncation and sampling of sums
    truncation_ratio_keep = K/K
    sampling_ratio_time_domain_keep = Nsymb/Nsymb
    sampling_ratio_subcarrier_domain_keep = K/K

    print('STEP 1: Parameter initialization is done.')

    # TRAINING - stage 1 ///////////////////////////////////////////////////////////////////////////////////////////////
    # In this stage, we do a phase noise free training to roughly optimize the beamformer
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # A. PHN-free dataset creation
    obj_dataset_train = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                                c, PHN_innovation_std, dataset_name, train_dataset_size)
    the_dataset_train, _, _, _, _ = obj_dataset_train.dataset_generator(mode="train", phase_noise="n")
    obj_dataset_test = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                               angular_spread_rad, wavelength, d, BATCHSIZE,
                                               phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                               c, PHN_innovation_std, dataset_name, test_dataset_size)
    the_dataset_test, _, _, _, _ = obj_dataset_test.dataset_generator(mode="test", phase_noise="n")
    print('STEP 2: Dataset creation is done.')

    obj_dataset_test_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                   angular_spread_rad, wavelength, d, BATCHSIZE,
                                                   phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts,
                                                   fc,
                                                   c, PHN_innovation_std, dataset_name, test_dataset_size)
    the_dataset_test_phn, H_tilde_0, H, Lambda_B, Lambda_U = obj_dataset_test_phn.dataset_generator(mode="test",
                                                                                                    phase_noise="y")
    # B. Creating H_tilde_0 for testing Sohrabi's method in matlab using the following matlab script FARAMARZ_A_MAIN_Final_with_combiner_in_presence_of_phase_noise
    H_tilde_0 = H_tilde_0.numpy()
    H = H.numpy()
    Lambda_B = Lambda_B.numpy()
    Lambda_U = Lambda_U.numpy()
    dictionary_for_saving_data_as_Matlab_vars = {"H_tilde_0": H_tilde_0, "H": H, "Lambda_B": Lambda_B,
                                                 "Lambda_U": Lambda_U}
    # print(H_tilde_0)
    sio.savemat("data_set_for_matlab.mat", dictionary_for_saving_data_as_Matlab_vars)
    print('data for testing Sohrabi\'s method in Matlab is created')


    # C. Loss function creation
    obj_loss_parallel_phase_noise_free = loss_parallel_phase_noise_free_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                                              SNR, P, N_c, N_scatterers,
                                                                              angular_spread_rad, wavelength, d,
                                                                              BATCHSIZE,
                                                                              phase_shift_stddiv)
    the_loss_function = obj_loss_parallel_phase_noise_free.ergodic_capacity

    print('3: Testing Sohrabi\'s method started.')
    obj_Sohrabi_s_method_tester = Sohrabi_s_method_tester_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                                N_scatterers, angular_spread_rad, wavelength,
                                                                d, BATCHSIZE, phase_shift_stddiv,
                                                                truncation_ratio_keep, Nsymb, Ts, fc, c,
                                                                dataset_for_testing_sohrabi,
                                                                test_dataset_size,
                                                                sampling_ratio_time_domain_keep,
                                                                sampling_ratio_subcarrier_domain_keep)
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

    sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/Matlab_projects/data8x8xk4xNrf4xNs1_85dbc_TestLutz2.mat", mdic)

    print("elapsed time of testing Sohrabi\' method = ", (end_time_4 - end_time_3), ' seconds')
    print("loss evaluation per sample = ", (end_time_4 - end_time_3)/test_dataset_size, ' seconds')

    #
    # end_time_5 = time.time()
    # C, C_samples_x_OFDM_index, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples = \
    #     obj_Sohrabi_s_method_tester.capacity_in_presence_of_phase_noise_Sohrabi()
    # end_time_6 = time.time()
    # print("loss evaluation per sample = ", (end_time_6 - end_time_5)/test_dataset_size, ' seconds')