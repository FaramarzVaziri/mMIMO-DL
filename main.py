
# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np
# tf.config.run_functions_eagerly(True)
# import matplotlib.pyplot as plt
# tf.distribute.Strategy

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> the device name: ',tf.config.list_physical_devices('GPU'))
if tf.test.gpu_device_name() == '/device:GPU:0':
    tf.device('/device:GPU:0')

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_model import CNN_model_class
from ML_model import ML_model_class
from Sohrabi_s_method_tester import Sohrabi_s_method_tester_class
from dataset_generator import dataset_generator_class
from loss_parallel_phase_noise_free import loss_parallel_phase_noise_free_class
from loss_parallel_phase_noised import paralle_loss_phase_noised_class
# tf.debugging.set_log_device_placement(True)


# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    print('The main code for running on this computer')

    print('tf version', tf.version.VERSION)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 1024  # int(input("No. train samples: "))
    test_dataset_size = 1024  # int(input("No. test samples: "))
    width_of_network = 1  # float(input("Network's width parameter: "))
    BATCHSIZE = 128  # int(input("batch size: "))
    L_rate = 1e-4  # float(input("inital lr: "))
    dropout_rate = .5  # float(input("dropout rate: "))
    precision_fixer = 1e-6  # float(input("precision fixer additive: "))
    # tensorboard_log_frequency = 1

    # The one that worked
    # # INPUTS ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    # train_dataset_size = 102400  # int(input("No. train samples: "))
    # test_dataset_size = 1024  # int(input("No. test samples: "))
    # width_of_network = 1  # float(input("Network's width parameter: "))
    # BATCHSIZE = 32  # int(input("batch size: "))
    # L_rate = 1e-4  # float(input("inital lr: "))
    # dropout_rate = .5  # float(input("dropout rate: "))
    # precision_fixer = 1e-6  # float(input("precision fixer additive: "))
    # tensorboard_log_frequency = 1
    #

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
    # tensorboard_log_frequency = 10

    PHN_innovation_std = np.sqrt( 4.0*np.pi**2*f_0**2 * 10**(L/10.) * Ts)
    print('PHN_innovation_std = ', PHN_innovation_std)

    # dataset_name = '/project/st-lampe-1/Faramarz/data/dataset/DS_for_py_for_training_ML.mat'
    # dataset_for_testing_sohrabi = '/project/st-lampe-1/Faramarz/data/dataset/DS_for_py_for_testing_Sohrabi.mat'

    dataset_name = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_training_ML.mat'
    dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'

    # Truncation and sampling of sums
    truncation_ratio_keep = 4/K
    sampling_ratio_time_domain_keep = 4/Nsymb
    sampling_ratio_subcarrier_domain_keep = 4/K

    print('STEP 1: Parameter initialization is done.')

    # GPU/CUDA info ////////////////////////////////////////////////////////////////////////////////////////////////////
    print('Some info about GPU:')
    from tensorflow.python.client import device_lib

    # print(device_lib.list_local_devices())
    gpu_available = tf.config.list_physical_devices('GPU')
    print('is GPU available: ', gpu_available)
    is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    print('is_cuda_gpu_available: ', is_cuda_gpu_available)
    is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3, 0))
    print('is_cuda_gpu_min_3: ', is_cuda_gpu_min_3)

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

    # B. ML model creation
    obj_CNN_model = CNN_model_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad,
                                    wavelength, d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep,
                                    Nsymb, Ts, fc, c, dataset_name, train_dataset_size, width_of_network, dropout_rate)
    the_CNN_model = obj_CNN_model.custom_CNN_plus_FC_with_functional_API()
    # print(the_ML_model)
    print('STEP 3: ML model creation is done.')

    # C. Loss function creation
    obj_loss_parallel_phase_noise_free = loss_parallel_phase_noise_free_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                                              SNR, P, N_c, N_scatterers,
                                                                              angular_spread_rad, wavelength, d,
                                                                              BATCHSIZE,
                                                                              phase_shift_stddiv)
    the_loss_function = obj_loss_parallel_phase_noise_free.ergodic_capacity

    # D. Training
    obj_ML_model = ML_model_class(model_dnn=the_CNN_model)
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=L_rate, clipnorm=1.)
    # optimizer = tf.keras.optimizers.SGD(learning_rate = L_rate , clipnorm=1.0) #0.0001
    tf.keras.utils.plot_model(the_CNN_model, show_shapes=True, show_layer_names=True, to_file='model.png')
    print(the_CNN_model.summary())
    obj_ML_model.compile(
        optimizer=optimizer_1,
        loss=the_loss_function,
        activation=obj_CNN_model.custom_actication,
        phase_noise='n')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity', factor=0.5, patience=4, min_lr=1e-12,
                                                     mode='min', verbose=1)
    log_dir = "logs_step1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "/project/st-lampe-1/Faramarz/data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=0, update_freq='epoch',profile_batch='5,6')


    print('STEP 4: Training in absence of phase noise has started.')
    start_time = time.time()
    obj_ML_model.fit(the_dataset_train, epochs=50, #10
                     validation_data=the_dataset_test, callbacks=[tensorboard_callback],
                     validation_batch_size=BATCHSIZE, verbose=1)

    end_time_1 = time.time()
    print("elapsed time of pre-training = ", (end_time_1 - start_time), ' seconds')

    # TRAINING - stage 2 ///////////////////////////////////////////////////////////////////////////////////////////////
    # In this stage, we do a phase noised training to accurately optimize the beamformer
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # A. PHN-free dataset creation
    obj_dataset_train_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                    angular_spread_rad, wavelength, d, BATCHSIZE,
                                                    phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts,
                                                    fc,
                                                    c, PHN_innovation_std, dataset_name, train_dataset_size)
    the_dataset_train_phn, _, _, _, _ = obj_dataset_train_phn.dataset_generator(mode="train",
                                                                                phase_noise="y")
    obj_dataset_test_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                   angular_spread_rad, wavelength, d, BATCHSIZE,
                                                   phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts,
                                                   fc,
                                                   c, PHN_innovation_std, dataset_name, test_dataset_size)
    the_dataset_test_phn, _, _, _, _ = obj_dataset_test_phn.dataset_generator(mode="test", phase_noise="y")
    print('STEP 5: Dataset creation is done.')

    # C. Loss function creation (sampled)
    obj_loss_parallel_phase_noised_approx = paralle_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                                     N_scatterers, angular_spread_rad, wavelength,
                                                                     d, BATCHSIZE, phase_shift_stddiv,
                                                                     truncation_ratio_keep, Nsymb,
                                                                     sampling_ratio_time_domain_keep,
                                                                     sampling_ratio_subcarrier_domain_keep)
    the_loss_function_phn_approx = obj_loss_parallel_phase_noised_approx.capacity_calculation_for_frame_for_batch

    # D. Transfer learning
    obj_ML_model_phn = ML_model_class(model_dnn=obj_ML_model.model_dnn)
    print('STEP 6: A new network has been initialized with the weights and biases of the previous network.')

    optimizer_2 = tf.keras.optimizers.Adam(learning_rate=L_rate/2, clipnorm=1.)
    # E. Training the new network with phase-noise perturbed loss function
    obj_ML_model_phn.compile(
        optimizer=optimizer_2,
        loss=the_loss_function_phn_approx,
        activation=obj_CNN_model.custom_actication,
        phase_noise='y')
    reduce_lrTF = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity', factor=0.5, patience=2, min_lr=1e-12, mode='min', verbose=1)

    log_dirTF = "logs_step_2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dirTF,
                                                     histogram_freq=1,
                                                     profile_batch='5,6')

    print('STEP 7: Training in presence of phase noise has started.')
    end_time_one_and_half = time.time()
    obj_ML_model_phn.fit(the_dataset_train_phn, epochs=10, #50
                         validation_data=the_dataset_test_phn,  callbacks=[tboard_callback],
                         validation_batch_size=BATCHSIZE, verbose=1)
    end_time_2 = time.time()
    print("elapsed time of stage-two training = ", (end_time_2 - end_time_one_and_half), ' seconds')
    
    # F. Evaluation
    obj_loss_parallel_phase_noised_accurate = paralle_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                                     N_scatterers, angular_spread_rad, wavelength,
                                                                     d, BATCHSIZE, phase_shift_stddiv,
                                                                     1, Nsymb,
                                                                     1,
                                                                     1)
    the_loss_function_phn_accurate = obj_loss_parallel_phase_noised_accurate.capacity_calculation_for_frame_for_batch

    obj_ML_model_phn.compile(
        optimizer=optimizer_2,
        loss=the_loss_function_phn_accurate,
        activation=obj_CNN_model.custom_actication,
        phase_noise='y')
    Capacity_simul = obj_ML_model_phn.evaluate(the_dataset_test_phn)
    print('monte-carlo simulation, C = ', Capacity_simul)


    # for tensorboard run this:
    # cd C:\Users\jabba\Google Drive\Main\Codes\ML_MIMO_new_project\PY_projects\convnet_transfer_learning_v1
    # tensorboard --logdir logs/fit/
