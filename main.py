# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np

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
    train_dataset_size = 10240
    test_dataset_size = 128
    eval_dataset_size = 128
    width_of_network = 5
    BATCHSIZE = 128
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



    obj_dataset_train = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                                c, PHN_innovation_std, dataset_name, train_dataset_size, 'train', 'no')
    the_dataset_train = obj_dataset_train.dataset_generator()
    obj_dataset_test = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                               angular_spread_rad, wavelength, d, BATCHSIZE,
                                               phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                               c, PHN_innovation_std, dataset_name, test_dataset_size, 'test', '-')
    the_dataset_test = obj_dataset_test.dataset_generator()
    print('-- Dataset creation is done.')

    obj_CNN_model = CNN_model_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad,
                                    wavelength, d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep,
                                    Nsymb, Ts, fc, c, dataset_name, train_dataset_size, width_of_network, dropout_rate)
    the_CNN_model = obj_CNN_model.custom_CNN_plus_FC_with_functional_API()
    print('-- CNN model is created')

    obj_loss_parallel_phase_noise_free = loss_parallel_phase_noise_free_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                                              SNR, P, N_c, N_scatterers,
                                                                              angular_spread_rad, wavelength, d,
                                                                              BATCHSIZE)
    the_loss_function = obj_loss_parallel_phase_noise_free.ergodic_capacity
    print('-- phase noise free loss function created')

    obj_capacity_metric = sequential_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, 1, Nsymb, 1, 1, 'eval')
    capacity_metric = obj_capacity_metric.capacity_forall_samples
    print('-- capacity metric with phase noise is created')

    obj_ML_model_pre_training = ML_model_class(the_CNN_model,N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                               N_scatterers, angular_spread_rad, wavelength, d, BATCHSIZE,
                                               phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c,
                                               PHN_innovation_std, dataset_name, eval_dataset_size, 'train',
                                               on_what_device, False)
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=L_rate, clipnorm=1.)
    # optimizer = tf.keras.optimizers.SGD(learning_rate = L_rate , clipnorm=1.0) #0.0001
    # tf.keras.utils.plot_model(the_CNN_model, show_shapes=True, show_layer_names=True, to_file='model.png')
    print(the_CNN_model.summary())
    obj_ML_model_pre_training.compile(
        optimizer=optimizer_1,
        loss=the_loss_function,
        activation=obj_CNN_model.custom_actication,
        phase_noise='n',
        metric_capacity_in_presence_of_phase_noise= capacity_metric)
    print('-- ML model is created')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity_train_loss', factor=0.1, patience=1, min_lr=1e-8,
                                                     mode='min', verbose=1)
    # log_dir = "logs_step1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    # log_dir = "/project/st-lampe-1/Faramarz/data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=0, update_freq='epoch',profile_batch='5,6')


    start_time = time.time()
    obj_ML_model_pre_training.fit(the_dataset_train,
                                  epochs=3, callbacks=[reduce_lr], validation_data=the_dataset_test, validation_freq=1, verbose=1) #

    end_time_1 = time.time()
    print("-- pre-training is done. (elapsed time = ", (end_time_1 - start_time), ' s)')


    print('-- Evaluation of the proposed pre-trained network has started')
    start_time_2 = time.time()

    C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples\
        = obj_ML_model_pre_training.evaluation_of_proposed_beamformer()

    # print('C: ', C)
    # print('capacity_sequence_in_frame_forall_samples : ', capacity_sequence_in_frame_forall_samples)
    # print('RX_forall_k_forall_OFDMs_forall_samples : ', RX_forall_k_forall_OFDMs_forall_samples)
    # print('RQ_forall_k_forall_OFDMs_forall_samples : ', RQ_forall_k_forall_OFDMs_forall_samples)

    C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
    RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
    RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

    mdic = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
            "L": L,
            'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
            'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
    if (on_what_device == 'cpu'):
        sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/Matlab_projects/evaluation_data_pre.mat", mdic)
    else:
        sio.savemat("/data/jabbarva/github_repo/mMIMO-DL/datasets/evaluation_data_pre.mat", mdic)

    end_time_2 = time.time()
    print('-- proposed pre-trained model is evaluated and data stored for matlab box-charts. (elapsed time =', (end_time_2 - start_time_2), ' seconds')



    # Evaluation of Sohrabi's method
    start_time_3 = time.time()
    C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples\
        = obj_ML_model_pre_training.evaluation_of_Sohrabis_beamformer()
    # print('C: ', C)
    # print('capacity_sequence_in_frame_forall_samples : ', capacity_sequence_in_frame_forall_samples)
    # print('RX_forall_k_forall_OFDMs_forall_samples : ', RX_forall_k_forall_OFDMs_forall_samples)
    # print('RQ_forall_k_forall_OFDMs_forall_samples : ', RQ_forall_k_forall_OFDMs_forall_samples)

    C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
    RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
    RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

    mdic = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
            "L": L,
            'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
            'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
    if (on_what_device == 'cpu'):
        sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/Matlab_projects/evaluation_data_Sohrabi.mat", mdic)
    else:
        sio.savemat("/data/jabbarva/github_repo/mMIMO-DL/datasets/evaluation_data_Sohrabi.mat", mdic)

    end_time_3 = time.time()
    print('-- Sohrabis method is evaluated and data stored for matlab box-charts. (elapsed time =', (end_time_3 - start_time_3), ' seconds')



    # post training ====================================================================================================
    # In this stage, we do a phase noised training to accurately optimize the beamformer
    print('-- Post-training started')
    obj_dataset_train_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                                c, PHN_innovation_std, dataset_name, train_dataset_size, 'train', 'yes')
    the_dataset_train_phn = obj_dataset_train_phn.dataset_generator()

    print('-- phase-noised Dataset is created.')

    obj_loss_phase_noised_approx = sequential_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                                      N_scatterers, angular_spread_rad, wavelength,
                                                                      d, BATCHSIZE, truncation_ratio_keep, Nsymb,
                                                                      sampling_ratio_time_domain_keep,
                                                                      sampling_ratio_subcarrier_domain_keep, 'train')

    the_loss_function_phn_approx = obj_loss_phase_noised_approx.capacity_forall_samples
    print('-- phase noised loss is created')

    # D. Transfer learning
    obj_ML_model_post_training = ML_model_class(the_CNN_model, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                               N_scatterers, angular_spread_rad, wavelength,
                                               d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                               c, PHN_innovation_std, dataset_name, eval_dataset_size, 'train',
                                               on_what_device, True)
    print('-- Transfer weights and biases is done')

    optimizer_2 = tf.keras.optimizers.Adam(learning_rate = L_rate / 2, clipnorm=1.)


    obj_ML_model_post_training.compile(
        optimizer=optimizer_2,
        loss=the_loss_function_phn_approx,
        activation=obj_CNN_model.custom_actication,
        phase_noise='y',
        metric_capacity_in_presence_of_phase_noise= capacity_metric)
    print('-- compiling the new ML model with new optimizer and phase noised loss is done')
    reduce_lrTF = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity_train_loss', factor=0.1, patience=1, min_lr=1e-8,
                                                     mode='min', verbose=1)


    # FIT
    print('-- Training in presence of phase noise has started.')
    start_time = time.time()
    obj_ML_model_post_training.fit(the_dataset_train_phn,
                                  epochs=2, callbacks=[reduce_lrTF], validation_data=the_dataset_test, validation_freq=1, verbose=1) #

    end_time_1 = time.time()
    print("-- elapsed time of post-training = ", (end_time_1 - start_time), ' seconds')


    print('-- Evaluation of the proposed post-trained network has started')
    start_time_4 = time.time()
    obj_sequential_loss_phase_noised_class_accurate = sequential_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, 1, Nsymb, 1, 1, 'eval')
    the_loss_function_phn_accurate = obj_sequential_loss_phase_noised_class_accurate.capacity_forall_samples

    C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples\
        = obj_ML_model_pre_training.evaluation_of_proposed_beamformer()

    # print('C: ', C)
    # print('capacity_sequence_in_frame_forall_samples : ', capacity_sequence_in_frame_forall_samples)
    # print('RX_forall_k_forall_OFDMs_forall_samples : ', RX_forall_k_forall_OFDMs_forall_samples)
    # print('RQ_forall_k_forall_OFDMs_forall_samples : ', RQ_forall_k_forall_OFDMs_forall_samples)

    C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
    RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
    RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

    mdic = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
            "L": L,
            'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
            'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
    if (on_what_device == 'cpu'):
        sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/Matlab_projects/evaluation_data_post.mat", mdic)
    else:
        sio.savemat("/data/jabbarva/github_repo/mMIMO-DL/datasets/evaluation_data_post.mat", mdic)

    end_time_4 = time.time()
    print('-- proposed pre-trained model is evaluated and data stored for matlab box-charts. (elapsed time =', (end_time_4 - start_time_4), ' seconds')

