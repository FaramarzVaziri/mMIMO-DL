import numpy as np
import scipy.io as sio
import tensorflow as tf

from CNN_model import CNN_model_class
from loss_parallel_phase_noised import paralle_loss_phase_noised_class


class Sohrabi_s_method_tester_class:
    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, mat_fname,
                 dataset_size, sampling_ratio_time_domain_keep, sampling_ratio_subcarrier_domain_keep):
        self.N_b_a = N_b_a
        self.N_b_rf = N_b_rf
        self.N_u_a = N_u_a
        self.N_u_rf = N_u_rf
        self.N_s = N_s
        self.K = K
        self.SNR = SNR
        self.P = P
        self.sigma2 = self.P / (10 ** (self.SNR / 10.))
        self.N_c = N_c
        self.N_scatterers = N_scatterers
        self.angular_spread_rad = angular_spread_rad  # 10deg
        self.wavelength = wavelength
        self.d = d
        self.BATCHSIZE = BATCHSIZE
        self.phase_shift_stddiv = phase_shift_stddiv
        self.truncation_ratio_keep = truncation_ratio_keep
        self.Nsymb = Nsymb
        self.Ts = Ts
        self.fc = fc
        self.c = c
        self.mat_fname = mat_fname
        self.dataset_size = dataset_size
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.sampling_ratio_subcarrier_domain_keep = sampling_ratio_subcarrier_domain_keep

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def capacity_in_presence_of_phase_noise_Sohrabi(self):
        obj_loss_parallel_phase_noised_2 = paralle_loss_phase_noised_class(self.N_b_a, self.N_b_rf, self.N_u_a,
                                                                           self.N_u_rf, self.N_s, self.K, self.SNR,
                                                                           self.P,
                                                                           self.N_c, self.N_scatterers,
                                                                           self.angular_spread_rad, self.wavelength,
                                                                           self.d, self.BATCHSIZE,
                                                                           self.phase_shift_stddiv,
                                                                           self.truncation_ratio_keep,
                                                                           self.Nsymb,
                                                                           self.sampling_ratio_time_domain_keep,
                                                                           self.sampling_ratio_subcarrier_domain_keep)

        the_loss_function_phn = obj_loss_parallel_phase_noised_2.capacity_calculation_for_frame_for_batch

        mat_contents = sio.loadmat(self.mat_fname)
        # No permutation is needed for the following data because they are not modified in Matlab and merely were passed
        # to the matlab code and came back here without any changes (so their sizes is also Ok)
        H = (mat_contents['H'])[0:self.dataset_size, :, :, :]
        Lambda_B = (mat_contents['Lambda_B'])[0:self.dataset_size, :, :, :, :]
        Lambda_U = (mat_contents['Lambda_U'])[0:self.dataset_size, :, :, :, :]

        # No permutation is needed for V_RF and W_RF as they do not have subcarrier dimension
        V_RF_Sohrabi_optimized = (mat_contents['V_RF_Sohrabi_optimized'])[0:self.dataset_size, :, :]
        W_RF_Sohrabi_optimized = (mat_contents['W_RF_Sohrabi_optimized'])[0:self.dataset_size, :, :]

        # The following data require permutation to bring k (subcarrier) to the second dimension
        V_D_Sohrabi_optimized = np.transpose(mat_contents['V_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                0:self.dataset_size, :, :, :]
        W_D_Sohrabi_optimized = np.transpose(mat_contents['W_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                0:self.dataset_size, :, :, :]
        # print(W_RF_Sohrabi_optimized)
        inputs0 = [V_D_Sohrabi_optimized, W_D_Sohrabi_optimized, H, V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, Lambda_B, Lambda_U]

        C, C_samples_x_OFDM_index, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples = the_loss_function_phn(inputs0)
        return C,\
               tf.squeeze(C_samples_x_OFDM_index),\
               tf.squeeze(RX_forall_k_forall_OFDMs_forall_samples), \
               tf.squeeze(RQ_forall_k_forall_OFDMs_forall_samples)
