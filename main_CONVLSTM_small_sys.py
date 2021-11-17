# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import h5py
import datetime
import timeit
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np
# import tensorboard
import os


try:
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))
  # strategy = tf.distribute.experimental.TPUStrategy(resolver)
  strategy = tf.distribute.TPUStrategy(resolver)

except ValueError:
  strategy = tf.distribute.get_strategy()


# tf.config.run_functions_eagerly(True)
# import matplotlib.pyplot as plt


# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_CONVLSTM import ResNet_model_class, Generic_2_ns_4_small_MIMOFDM_CNN_class, Generic_2_ns_4_large_MIMOFDM_CNN_class, Specialized_2_ns_4_small_MIMOFDM_CNN_class, Specialized_2_ns_4_large_MIMOFDM_CNN_class # this is the best currently, it has repetitions and no shortcut
from ML_model_CONVLSTM import ML_model_class
from dataset_generator import dataset_generator_class
from loss_phase_noise_free import loss_phase_noise_free_class
from loss_phase_noised_CONVLSTM import loss_phase_noised_class


def training_metadata_writer(file_name, training_metadata):
    try:
        training_metadata_records_loaded = sio.loadmat(file_name)
        training_metadata_records_new = np.concatenate((training_metadata_records_loaded['training_metadata']
                                                        , [training_metadata])
                                                       , axis=0)
        training_metadata_dict = {'training_metadata': training_metadata_records_new}
        sio.savemat(file_name, training_metadata_dict)
    except:
        training_metadata_dict = {'training_metadata': training_metadata}
        sio.savemat(file_name, training_metadata_dict)

# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    generic_part_trainable = True
    specialized_part_trainable = True
    impl = 'map_fn'
    impl_phn_free = 'map_fn'

    epochs_pre = 10
    epochs_post = 10
    val_freq_pre = 1
    val_freq_post = 1

    # pre
    do_pre_train = 'yes'
    save_pre = 'yes'
    evaluate_pre = 'yes'

    # post
    load_trained_best_model = 'no'
    do_post_train = 'yes'
    save_post = 'yes'
    evaluate_post = 'yes'

    evaluate_sohrabi = 'no'
    # KPIs
    record_loss = 'yes'
    record_metadata = 'yes'

    # ML Setup /////////////////////////////////////////////////////////////////////////////////////////////////////////
    train_dataset_size = 10240
    train_data_fragment_size = train_dataset_size
    train_dataset_size_post = 10240
    train_data_fragment_size_post = train_dataset_size_post
    test_dataset_size = 128
    test_data_fragment_size = test_dataset_size
    eval_dataset_size = test_dataset_size
    eval_data_fragment_size = eval_dataset_size
    BATCHSIZE = 128
    L_rate = 1e-3
    dropout_rate = 0.0

    # DNN setting
    convolutional_kernels = 5
    convolutional_filters = 12
    convolutional_strides = 1
    convolutional_dilation = 1
    subcarrier_strides = 1
    N_b_a_strides = 1
    N_u_a_strides = 1

    # MIMO-OFDM setup //////////////////////////////////////////////////////////////////////////////////////////////////
    N_b_a = 4
    N_b_rf = 2
    N_b_o = N_b_rf
    N_u_a = 4
    N_u_rf = 2
    N_u_o = N_u_rf
    N_s = 1
    K = 4
    PTRS_seperation = round(K/2)
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

    phase_noise_power_augmentation_factor = np.sqrt(1.)
    PHN_innovation_std = np.sqrt(1024/K)*np.sqrt(4.0 * np.pi ** 2 * f_0 ** 2 * 10 ** (L / 10.) * Ts)
    print('-- PHN_innovation_std = ', PHN_innovation_std)

    dataset_name = 'Dataset_K4_Na4_rf2/DS.mat'
    dataset_for_testing_sohrabi = 'DS_for_py_for_testing_Sohrabi.mat'


    # Truncation and sampling of sums
    truncation_ratio_keep = 4 / K
    number_of_OFDM_symbols_considered = 50
    sampling_ratio_time_domain_keep = number_of_OFDM_symbols_considered / Nsymb
    sampling_ratio_subcarrier_domain_keep = 4 / K

    print('-- Parameter initialization is done.')

    obj_dataset_train = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                                c, PHN_innovation_std*phase_noise_power_augmentation_factor, dataset_name, train_dataset_size, train_data_fragment_size, 'train', 'no')
    the_dataset_train = obj_dataset_train.dataset_generator()
    print('-- cardinality of the train DS: ', tf.data.experimental.cardinality(the_dataset_train))
    obj_dataset_test = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                               angular_spread_rad, wavelength, d, BATCHSIZE,
                                               phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                               c, PHN_innovation_std, dataset_name, test_dataset_size, test_data_fragment_size, 'test', '-')
    the_dataset_test = obj_dataset_test.dataset_generator()
    print('-- Dataset creation is done.')

    obj_neural_net_model = ResNet_model_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, SNR, P, N_c, N_scatterers, angular_spread_rad,
                                              wavelength, d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep,
                                              Nsymb, sampling_ratio_time_domain_keep, Ts, fc, c, dataset_name, train_dataset_size, dropout_rate,
                                              convolutional_kernels, convolutional_filters, convolutional_strides, convolutional_dilation,
                                              subcarrier_strides, N_b_a_strides, N_u_a_strides,
                                              generic_part_trainable, specialized_part_trainable)
    the_model_tx = obj_neural_net_model.resnet_4_small_MIMOFDM_transmitter()
    print('-- TX resnet model is created')
    the_model_rx = obj_neural_net_model.resnet_4_small_MIMOFDM_receiver()
    print('-- RX resnet model is created')

    obj_loss_phase_noise_free_class = loss_phase_noise_free_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                                  SNR, P, N_c, N_scatterers,
                                                                  angular_spread_rad, wavelength, d,
                                                                  BATCHSIZE,Nsymb, sampling_ratio_time_domain_keep,
                                                                  impl_phn_free, sampling_ratio_subcarrier_domain_keep)
    the_loss_function = obj_loss_phase_noise_free_class.capacity_forall_samples
    print('-- phase noise free loss function created')

    obj_capacity_metric = loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                                                  d, BATCHSIZE, 1, Nsymb, sampling_ratio_time_domain_keep, 1, 'eval', impl)
    capacity_metric = obj_capacity_metric.capacity_forall_samples
    print('-- capacity metric with phase noise is created')


    obj_ML_model_pre_training = ML_model_class(the_model_tx, the_model_rx, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, SNR, P, N_c,
                                               N_scatterers, angular_spread_rad, wavelength, d, BATCHSIZE,
                                               phase_shift_stddiv, truncation_ratio_keep, sampling_ratio_time_domain_keep, Nsymb, Ts, fc, c,
                                               PHN_innovation_std, dataset_name, eval_dataset_size, 'train', False, number_of_OFDM_symbols_considered)

    gradient_norm_clipper = 1.
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=L_rate)#, clipnorm = gradient_norm_clipper)
    # tx
    tf.keras.utils.plot_model(the_model_tx, show_shapes=True, show_layer_names=True, to_file='cnn_tx.png')
    the_model_tx.summary()
    n_params_tx = the_model_tx.count_params()
    n_layers_tx = len(the_model_tx.layers)
    # rx
    tf.keras.utils.plot_model(the_model_rx, show_shapes=True, show_layer_names=True, to_file='cnn_rx.png')
    the_model_rx.summary()
    n_params_rx = the_model_rx.count_params()
    n_layers_rx = len(the_model_rx.layers)



    obj_ML_model_pre_training.compile(
        optimizer=optimizer_1,
        loss=the_loss_function,
        activation_TX=obj_neural_net_model.custom_actication_transmitter,
        activation_RX=obj_neural_net_model.custom_actication_receiver,
        metric_capacity_in_presence_of_phase_noise = capacity_metric)

    print('train_dataset_size=', train_dataset_size,
          ' train_dataset_size_post=', train_dataset_size_post,
          ' test_dataset_size=', test_dataset_size,
          ' eval_dataset_size=', eval_dataset_size,
          'BATCHSIZE=', BATCHSIZE,
          'n_layers_tx', n_layers_tx,
          'n_layers_rx', n_layers_rx,
          ' L_rate=', L_rate,
          'convolutional_kernels', convolutional_kernels)

    print('-- ML model is created')

    ReduceLROnPlateau_decay_rate = 0.1
    ReduceLROnPlateau_patience = 1
    ReduceLROnPlateau_min_lr = 1e-8
    reduce_lr_pre = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity_train_loss',
                                                     factor= ReduceLROnPlateau_decay_rate, patience= ReduceLROnPlateau_patience, min_lr= ReduceLROnPlateau_min_lr,
                                                     mode='min', verbose=1)

    # checkpoint_path_pre = "checkpoints/pre/best_model_weights_only/cp.ckpt"
    # checkpoint_dir_pre = os.path.dirname(checkpoint_path_pre)
    # checkpoint_callback_pre = tf.keras.callbacks.ModelCheckpoint(
    #     filepath= checkpoint_dir_pre, monitor='neg_capacity_train_loss', verbose=1, save_best_only=True,
    #     save_weights_only=True, mode='min', save_freq='epoch',
    #     options=None
    # )
    # log_dir = "/content/gdrive/MyDrive/Main/Codes/ML_MIMO_new_project/PY_projects/convnet_transfer_learning_v1/logs/fit/" + \
    #           datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir) # , profile_batch=2


    if (load_trained_best_model == 'yes'):
        # best_model_post = tf.train.latest_checkpoint("checkpoints/post")
        # obj_ML_model_post_training.load_weights(best_model_post)
        obj_ML_model_pre_training.built = True
        obj_ML_model_pre_training.load_weights('saved_model_weights/my_model.h5')

    if (do_pre_train == 'yes'):
        print('-- Pre-training started')
        start_time = time.time()
        h_pre = obj_ML_model_pre_training.fit(the_dataset_train,
                                       epochs=epochs_pre, callbacks=[reduce_lr_pre]
                                       ,validation_data=the_dataset_test,
                                       validation_freq= val_freq_pre,
                                       verbose=1)
        if (save_pre == 'yes'):
            obj_ML_model_pre_training.save_weights('saved_model_weights/pre/my_model.h5')
            obj_ML_model_pre_training.save_weights('saved_model_weights/my_model.h5')
            print("-- pre-trained network is saved.")

        end_time_1 = time.time()
        print("-- pre-training is done. (elapsed time = ", (end_time_1 - start_time), ' s)')


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    if (evaluate_pre == 'yes'):
        print('-- Evaluation of the proposed pre-trained network has started')
        start_time_2 = time.time()

        C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples\
            = obj_ML_model_pre_training.evaluation_of_proposed_beamformer()

        C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
        RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
        RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

        mdic = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
                "L": L,
                'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
                'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
        sio.savemat("evaluation_data_pre.mat", mdic)
        end_time_2 = time.time()
        print('-- proposed pre-trained model is evaluated and data stored for matlab box-charts. (elapsed time =', (end_time_2 - start_time_2), ' seconds')

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    if (evaluate_sohrabi == 'yes'):
        print('-- Evaluation of Sohrabis method has started')
        start_time_3 = time.time()

        C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples \
            = obj_ML_model_pre_training.evaluation_of_Sohrabis_beamformer()

        C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
        RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
        RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

        mdic_soh = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
                "L": L,
                'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
                'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
        sio.savemat("evaluation_data_Sohrabi.mat", mdic_soh)

        end_time_3 = time.time()
        print('-- Sohrabis method is evaluated and data stored for matlab box-charts. (elapsed time =',
              (end_time_3 - start_time_3), ' seconds')

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================

    # post training ====================================================================================================
    # In this stage, we do a phase noised training to accurately optimize the beamformer

    obj_dataset_train_phn = dataset_generator_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers,
                                                angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc,
                                                c, phase_noise_power_augmentation_factor*PHN_innovation_std, dataset_name, train_dataset_size_post, train_data_fragment_size_post, 'train', 'yes')
    the_dataset_train_phn = obj_dataset_train_phn.dataset_generator()

    print('-- phase-noised Dataset is created.')

    obj_loss_phase_noised_approx = loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
                                                           N_scatterers, angular_spread_rad, wavelength,
                                                           d, BATCHSIZE, truncation_ratio_keep, Nsymb,
                                                           sampling_ratio_time_domain_keep,
                                                           sampling_ratio_subcarrier_domain_keep, 'train', impl)

    the_loss_function_phn_approx = obj_loss_phase_noised_approx.capacity_forall_samples
    print('-- phase noised loss is created')

    # D. Transfer learning
    obj_ML_model_post_training = ML_model_class(the_model_tx, the_model_rx, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K,
                                                PTRS_seperation, SNR, P, N_c,
                                                N_scatterers, angular_spread_rad, wavelength, d, BATCHSIZE,
                                                phase_shift_stddiv, truncation_ratio_keep, sampling_ratio_time_domain_keep, Nsymb, Ts, fc, c,
                                                PHN_innovation_std, dataset_name, eval_dataset_size, 'train', True, number_of_OFDM_symbols_considered)


    print('-- Transfer weights and biases is done')

    optimizer_2 = tf.keras.optimizers.Adam(learning_rate = L_rate, clipnorm = gradient_norm_clipper)
    obj_ML_model_post_training.compile(
        optimizer=optimizer_2,
        loss=the_loss_function_phn_approx,
        activation_TX=obj_neural_net_model.custom_actication_transmitter,
        activation_RX=obj_neural_net_model.custom_actication_receiver,
        metric_capacity_in_presence_of_phase_noise=capacity_metric)


    print('-- compiling the new ML model with new optimizer and phase noised loss is done')
    reduce_lrTL = tf.keras.callbacks.ReduceLROnPlateau(monitor='neg_capacity_train_loss', factor= ReduceLROnPlateau_decay_rate,
                                                       patience=ReduceLROnPlateau_patience, min_lr=ReduceLROnPlateau_min_lr, mode='min', verbose=1)

    # checkpoint_path_post = "checkpoints/post/best_model_weights_only/cp.ckpt"
    # checkpoint_dir_post = os.path.dirname(checkpoint_path_post)
    # checkpoint_callback_post = tf.keras.callbacks.ModelCheckpoint(
    #     checkpoint_dir_post, monitor='neg_capacity_train_loss', verbose=1, save_best_only=True,
    #     save_weights_only=True, mode='min', save_freq='epoch',
    #     options=None
    # )
    # log_dir = "/content/gdrive/MyDrive/Main/Codes/ML_MIMO_new_project/PY_projects/convnet_transfer_learning_v1/logs/fit/" + \
    #           datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir) # , profile_batch=2

    if (do_post_train == 'yes'):
        print('-- Training in presence of phase noise has started.')
        start_time_4 = time.time()
        h_post = obj_ML_model_post_training.fit(the_dataset_train_phn,
                                                epochs=epochs_post,
                                                callbacks=[reduce_lrTL]
                                                 ,validation_data=the_dataset_test,
                                                validation_freq= val_freq_post,
                                                 verbose=1) #
        if (save_post == 'yes'):
            obj_ML_model_post_training.save_weights('saved_model_weights/post/my_model.h5')
            obj_ML_model_post_training.save_weights('saved_model_weights/my_model.h5')
            print("-- post-trained network is saved.")

        end_time_4 = time.time()
        print("-- elapsed time of post-training = ", (end_time_4 - start_time_4), ' seconds')

    if (record_loss == 'yes'):
        try:
            mdic_losses = {"tr_loss":  np.concatenate([h_pre.history['neg_capacity_train_loss'], h_post.history['neg_capacity_train_loss']]),
                                'ts_loss': np.concatenate([h_pre.history['val_neg_capacity_test_loss'], h_post.history['val_neg_capacity_test_loss']]),
                                'metric': np.concatenate([h_pre.history['val_neg_capacity_performance_metric'], h_post.history['val_neg_capacity_performance_metric']]),
                                'TL_start': epochs_pre}
        except:
            mdic_losses = {"tr_loss": h_post.history['neg_capacity_train_loss'],
                                'ts_loss': h_post.history['val_neg_capacity_test_loss'],
                                'metric': h_post.history['val_neg_capacity_performance_metric'],
                                'TL_start': epochs_pre}

        sio.savemat("loss_data.mat", mdic_losses)


    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    if (evaluate_post == 'yes'):
        print('-- Evaluation of the proposed post-trained network has started')
        start_time_5 = time.time()
        obj_sequential_loss_phase_noised_class_accurate = loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                                                                                  d, BATCHSIZE, 1, Nsymb, sampling_ratio_time_domain_keep, 1, 'eval', impl)
        the_loss_function_phn_accurate = obj_sequential_loss_phase_noised_class_accurate.capacity_forall_samples

        C, capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples\
            = obj_ML_model_pre_training.evaluation_of_proposed_beamformer()

        C_samples_x_OFDM_index = capacity_sequence_in_frame_forall_samples.numpy()
        RX_forall_k_forall_OFDMs_forall_samples = RX_forall_k_forall_OFDMs_forall_samples.numpy()
        RQ_forall_k_forall_OFDMs_forall_samples = RQ_forall_k_forall_OFDMs_forall_samples.numpy()

        mdic_post = {"C_samples_x_OFDM_index": C_samples_x_OFDM_index,
                "L": L,
                'RX_forall_k_forall_OFDMs_forall_samples': RX_forall_k_forall_OFDMs_forall_samples,
                'RQ_forall_k_forall_OFDMs_forall_samples': RQ_forall_k_forall_OFDMs_forall_samples}
        sio.savemat("evaluation_data_post.mat", mdic_post)

        end_time_5 = time.time()
        print('-- proposed pre-trained model is evaluated and data stored for matlab box-charts. (elapsed time =', (end_time_5 - start_time_5), ' seconds')

    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # ==================================================================================================================
    # parallel coordinate plots
    if (record_metadata == 'yes'):
        training_metadata = np.array(
            [train_dataset_size,
             train_dataset_size_post,
             BATCHSIZE,
             L_rate,
             gradient_norm_clipper,
             ReduceLROnPlateau_decay_rate,
             ReduceLROnPlateau_patience,
             ReduceLROnPlateau_min_lr,
             dropout_rate,
             convolutional_kernels,
             convolutional_filters,
             convolutional_strides,
             n_params_tx,
             n_params_rx,
             n_layers_tx,
             n_layers_rx,
             epochs_pre,
             epochs_post,
             -h_pre.history['neg_capacity_train_loss'][-1],
             -h_pre.history['val_neg_capacity_test_loss'][-1],
             -h_pre.history['val_neg_capacity_performance_metric'][-1],
             -h_post.history['neg_capacity_train_loss'][-1],
             -h_post.history['val_neg_capacity_test_loss'][-1],
             -h_post.history['val_neg_capacity_performance_metric'][-1]])

        training_metadata_writer(file_name='training_metadata_records.mat', training_metadata = training_metadata)

        print('-- Parralel coordinate plot info is written to the disk')