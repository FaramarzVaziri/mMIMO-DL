# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import warnings

import tensorflow as tf
import numpy as np
import timeit
import datetime, os
import time
from keras.utils.vis_utils import plot_model
import sklearn
from datetime import datetime
from packaging import version
import matplotlib.pyplot as plt

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from loss_serial import serial_loss_class
from loss_parallel import parallel_loss_class
from loss_parallel_phase_noised import paralle_loss_phase_noised_class

# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main_for_chart_box__':
    print('tf version', tf.version.VERSION)

    # parameters:
    N_b_a = 1
    N_b_rf = 1
    N_b_o = N_b_rf
    N_u_a = 1
    N_u_rf = 1
    N_u_o = N_u_rf
    N_s = 1
    K = 32
    SNR = 20.
    P = 1.
    sigma2 = P / (10 ** (SNR / 10.))
    N_c = 5
    N_scatterers = 10
    angular_spread_rad = 0.1745  # 10deg
    wavelength = 1.
    d = .5
    phi_c = .01
    BATCHSIZE = 1
    phase_shift_stddiv = 0.0
    truncation_ratio_keep = .9
    Nsymb = 14

    # Initialization ///////////////////////////////////////////////////////////////////////////////////////////////////
    V_D = 0.5 * tf.complex(tf.random.normal(shape=[BATCHSIZE, K, N_b_rf, N_s], dtype=tf.float32),
                           tf.random.normal(shape=[BATCHSIZE, K, N_b_rf, N_s], dtype=tf.float32))

    V_RF = 0.5 * tf.complex(tf.random.normal(shape=[BATCHSIZE, N_b_a, N_b_rf], dtype=tf.float32),
                            tf.random.normal(shape=[BATCHSIZE, N_b_a, N_b_rf], dtype=tf.float32))

    W_RF = 0.5 * tf.complex(tf.random.normal(shape=[BATCHSIZE, N_u_a, N_u_rf], dtype=tf.float32),
                            tf.random.normal(shape=[BATCHSIZE, N_u_a, N_u_rf], dtype=tf.float32))

    W_D = 0.5 * tf.complex(tf.random.normal(shape=[BATCHSIZE, K, N_u_rf, N_s], dtype=tf.float32),
                           tf.random.normal(shape=[BATCHSIZE, K, N_u_rf, N_s], dtype=tf.float32))

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # PHASE NOISE GENERATION ///////////////////////////////////////////////////////////////////////////////////////////
    # these three functions take care of repeting the phase noise for the antennas of the same oscillator
    @tf.function
    def PHN_forall_RF_per_k_per_sample(Theta_forall_RF_per_k_per_sample):
        return tf.linalg.diag(
            tf.repeat(Theta_forall_RF_per_k_per_sample, repeats=tf.cast(N_b_a / N_b_rf, dtype=tf.int32), axis=0))

    @tf.function
    def PHN_forall_RF_foall_k_per_sample(Theta_forall_RF_foall_k_per_sample):
        return tf.map_fn(PHN_forall_RF_per_k_per_sample, Theta_forall_RF_foall_k_per_sample)

    @tf.function
    def PHN_foall_k_forall_RF_forall_samples(Theta_forall_samples):
        return tf.map_fn(PHN_forall_RF_foall_k_per_sample, Theta_forall_samples)


    # based on Elena's email
    # def Wiener_phase_noise_generator_Elena(n_samples, K, N_b_rf):
    #     Fsamp_cmn = 32.72e6
    #     Tsamp_cmn = 1 / Fsamp_cmn
    #     pn_level_dBc = -90  # Phase noise level in dBc/Hz
    #     pn_f_offset = 200e3  # Offset frequency in Hz
    #     pn_std_sam = 10 ** (pn_level_dBc / 20) * 2 * np.pi * pn_f_offset / np.sqrt(1 / Tsamp_cmn)
    #     PNsamps = np.float32(np.cumsum(pn_std_sam * np.random.normal(size=n_samples * K * N_b_rf)))
    #     return PNsamps


    # the following phase noise is based on R. Zhang, B. Shim and H. Zhao, "Downlink Compressive Channel Estimation With Phase Noise in Massive MIMO Systems," in IEEE Transactions on Communications, vol. 68, no. 9, pp. 5534-5548, Sept. 2020, doi: 10.1109/TCOMM.2020.2998141.
    def Wiener_phase_noise_generator_Ruoyu(BATCHSIZE, Nsymb, K, N_rf):
        Ts = 1. / (200e6)
        fc = 22.0e9
        c = 4.7e-17  #
        pn_std_sam = 2 * np.pi * fc * np.sqrt(c * Ts)
        PNsamps = np.float32(np.cumsum(np.random.normal(loc=0., scale=pn_std_sam, size=BATCHSIZE* Nsymb * K * N_rf)))
        PNsamps_cplx = (tf.complex(tf.cos(PNsamps), tf.sin(PNsamps))) / K
        PNsamps_cplx_K_Nrf = tf.reshape(PNsamps_cplx, shape=[BATCHSIZE, Nsymb, N_rf, K])
        DFT_PNsamps_cplx_K_Nrf = tf.signal.fft(PNsamps_cplx_K_Nrf)
        trans_DFT_PNsamps_cplx_K_Nrf = tf.transpose(DFT_PNsamps_cplx_K_Nrf, perm=[0, 1, 3, 2])  # batch, symb, k, rf
        return PNsamps_cplx_K_Nrf, trans_DFT_PNsamps_cplx_K_Nrf


    relative_error = []

    for i in range(10):

        dummy1, PHN_B_DFT_domain_samples_K_Nrf_train = Wiener_phase_noise_generator_Ruoyu(BATCHSIZE, Nsymb, K, N_b_rf)
        Lambda_B = PHN_foall_k_forall_RF_forall_samples(PHN_B_DFT_domain_samples_K_Nrf_train)
        # PHN_B_dataset_train = tf.data.Dataset.from_tensor_slices(Lambda_B_measured_PHN_train)

        # plt.stem(np.abs(PHN_B_DFT_domain_samples_K_Nrf_train[0,0,:,0]))
        # plt.show()

        # UE
        dummy2, PHN_U_DFT_domain_samples_K_Nrf_train = Wiener_phase_noise_generator_Ruoyu(BATCHSIZE, Nsymb, K, N_u_rf)
        Lambda_U = PHN_foall_k_forall_RF_forall_samples(PHN_U_DFT_domain_samples_K_Nrf_train)
        # PHN_U_dataset_train = tf.data.Dataset.from_tensor_slices(Lambda_U_measured_PHN_train)



        # Lambda_B = tf.tile([tf.concat(
        #     [[tf.eye(N_b_a, dtype=tf.complex64)], tf.zeros(shape=[K - 1, N_b_a, N_b_a], dtype=tf.complex64)], axis=0)],
        #                    multiples=[BATCHSIZE, 1, 1, 1])
        H = tf.random.normal(shape=[BATCHSIZE, K, N_u_a, N_b_a, 2], dtype=tf.float32)

        # Lambda_U = tf.tile([tf.concat(
        #     [[tf.eye(N_u_a, dtype=tf.complex64)], tf.zeros(shape=[K - 1, N_u_a, N_u_a], dtype=tf.complex64)], axis=0)],
        #                    multiples=[BATCHSIZE, 1, 1, 1])

        inn = [V_D, W_D, H, V_RF, W_RF]
        inn_PHN = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]


        # testing serial loss's speed //////////////////////////////////////////////////////////////////////////////////////
        obj_serial = serial_loss_class(N_b_a,N_b_rf,N_u_a,N_u_rf,N_s,K,SNR,P,N_c,N_scatterers,angular_spread_rad,wavelength,d,BATCHSIZE,phase_shift_stddiv)
        # print(obj_serial.loss_func_custom_serial(inn))

        starttime = timeit.default_timer()
        obj_serial.loss_func_custom_serial(inn)
        # print("Elapsed time for serial :", timeit.default_timer() - starttime)



        tf.config.run_functions_eagerly(False)

        # testing parallel loss's speed ////////////////////////////////////////////////////////////////////////////////////
        obj_parallel = parallel_loss_class(N_b_a,N_b_rf,N_u_a,N_u_rf,N_s,K,SNR,P,N_c,N_scatterers,angular_spread_rad,wavelength,d,BATCHSIZE,phase_shift_stddiv)
        # print('loss w/o PHN = ', obj_parallel.loss_func_custom_parallel(inn))

        starttime = timeit.default_timer()
        obj_parallel.loss_func_custom_parallel(inn)
        # print("elappsed time = ", timeit.default_timer() - starttime)

        # testing parallel loss with phase noise speed /////////////////////////////////////////////////////////////////////
        obj_parallel_phase_noised_with_approx = paralle_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR,
                                                                                P, N_c, N_scatterers,
                                                                                angular_spread_rad, wavelength, d,
                                                                                BATCHSIZE, phase_shift_stddiv,
                                                                                truncation_ratio_keep=truncation_ratio_keep,
                                                                                Nsymb=Nsymb)
        C_PHN_with_approx = obj_parallel_phase_noised_with_approx.capacity_calculation_for_frame_for_batch(inn_PHN)
        # print('loss with PHN approx = ', C_PHN_with_approx)

        starttime = timeit.default_timer()
        obj_parallel_phase_noised_with_approx.capacity_calculation_for_frame_for_batch(inn_PHN)
        # print("elappsed time = ", timeit.default_timer() - starttime)

        obj_parallel_phase_noised_without_approx = paralle_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR,
                                                                                P, N_c, N_scatterers,
                                                                                angular_spread_rad, wavelength, d,
                                                                                BATCHSIZE, phase_shift_stddiv,
                                                                                truncation_ratio_keep=0.0,
                                                                                Nsymb=Nsymb)
        C_PHN_without_approx = obj_parallel_phase_noised_without_approx.capacity_calculation_for_frame_for_batch(inn_PHN)
        # print('loss with PHN = ', C_PHN_without_approx)

        # starttime = timeit.default_timer()
        # obj_parallel_phase_noised_without_approx.capacity_calculation_for_frame_for_batch(inn_PHN)
        # print("elappsed time = ", timeit.default_timer() - starttime)

        r = 100.*tf.math.abs((C_PHN_with_approx-C_PHN_without_approx)/C_PHN_without_approx)
        # print('relative error = ', r)

        relative_error.append(r)

    print(tf.stack(relative_error, axis=0))

    #
    #
    # #
    # # obj_parallel_phase_noised = paralle_loss_phase_noised_class(N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c,
    # #                                                             N_scatterers,
    # #                                                             angular_spread_rad, wavelength, d, BATCHSIZE,
    # #                                                             phase_shift_stddiv, truncation_on=False,
    # #                                                             truncation_ratio_keep=0.0)
    # #
    # # # fft_samps = tf.constant(range(K), dtype=tf.complex64)
    # # # fft_samps = tf.constant([4,3,2,1,0,1,2,3,4], dtype=tf.complex64)
    # #
    # fft_samps = tf.constant(tf.signal.fft(PHN_B_samples_train, K))
    # # print('PHN_B_samples_train', PHN_B_samples_train)
    # dd = 0
    # #
    # # PLOT FT
    # # fft_samps_shifted1 = obj_parallel_phase_noised2.cyclical_shift(fft_samps, dd, flip=True)
    # # plt.stem(tf.math.abs(fft_samps_shifted1))
    # # plt.ylim([0, 4])
    # # plt.show()
    # #
    # # fft_samps_cyc_shifted_approximated = tf.multiply(
    # #     obj_parallel_phase_noised.cyclical_shift(fft_samps, dd, flip=True)
    # #     , tf.cast(obj_parallel_phase_noised.non_zero_element_finder_ft(dd, .5), dtype=tf.complex64))
    # # plt.stem(tf.math.abs(fft_samps_cyc_shifted_approximated))
    # # # plt.ylim([0, 4])
    # # plt.show()
    # #
    # # # PLOT FF
    # fft_samps_shifted2 = obj_parallel_phase_noised2.cyclical_shift(fft_samps, dd, flip=False)
    # plt.stem(tf.math.abs(fft_samps_shifted2))
    # # plt.ylim([0, 4])
    # plt.show()
    # #
    # fft_samps_cyc_shifted_approximated = tf.multiply(
    #     obj_parallel_phase_noised2.cyclical_shift(fft_samps, dd, flip=False)
    #     , tf.cast(obj_parallel_phase_noised2.non_zero_element_finder_for_H_tilde_ff(dd, truncation_ratio_keep), dtype=tf.complex64))
    # plt.stem(tf.math.abs(fft_samps_cyc_shifted_approximated))
    # # plt.ylim([0, 4])
    # plt.show()