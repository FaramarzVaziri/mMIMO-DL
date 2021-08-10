# DATASET GENERATION ///////////////////////////////////////////////////////////////////////////////////////////////
# every sample contains the channel and phase noise information about Nsymb consecutive OFDM symbols
# Since the channel model is block fading, H[k] itself remains the same during the frame and since H_tilde_ns[k] is generated in the loss function, there is no need to have it in the dataset
# However, H_tilde_0[k] needs to be generated as it is the input of the CNN
# Also, Lambda_B/U for all ns should be generated for one sample
# In summary: one sample is (H[k] forall k, Lambda_B_ns[k] forall k forall ns, Lambda_U_ns[k] forall k forall ns, H_tilde_0[k] forall k)

import numpy as np
import scipy.io as sio
import tensorflow as tf
from os.path import dirname, join as pjoin


# if tf.test.gpu_device_name() == '/device:GPU:0':
#   tf.device('/device:GPU:0')

class dataset_generator_class:

    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, PHN_innovation_std,
                 mat_fname, dataset_size, mode, phase_noise):
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
        self.PHN_innovation_std = PHN_innovation_std
        self.mode = mode
        self.phase_noise = phase_noise

    # PHASE NOISE GENERATION////////////////////////////////////////////////////////////////////////////////////////////
    # these three functions take care of repeating the phase noise for the antennas of the same oscillator

    
    def PHN_forall_RF(self, theta):
        # print('should be N_rf but is: ', theta.shape)
        T0 = tf.linalg.diag(tf.repeat(theta, repeats=tf.cast(self.N_b_a / self.N_b_rf, dtype=tf.int32),
                                      axis=0))
        return T0

    
    def PHN_forall_RF_forall_K(self, theta):
        # print('should be K*N_rf but is: ', theta.shape)
        return tf.map_fn(self.PHN_forall_RF, theta)

    
    def PHN_forall_RF_forall_K_forall_symbols(self, theta):
        # print('should be Nsymb*K*N_rf but is: ', theta.shape)
        return tf.map_fn(self.PHN_forall_RF_forall_K, theta)

    
    def PHN_forall_RF_forall_K_forall_symbols_forall_samples(self, theta):
        # print('should be dataset_size*Nsymb*K*N_rf but is: ', theta.shape)
        return tf.map_fn(self.PHN_forall_RF_forall_K_forall_symbols, theta)

    
    def Wiener_phase_noise_generator_Ruoyu_for_one_frame_forall_RF(self, Nrf):
        if (self.mode == 'train' and self.phase_noise == 'no'):
            N_symbols = 1
        elif(self.mode == 'train' and self.phase_noise == 'yes'):
            N_symbols = self.Nsymb
        else:
            N_symbols = self.Nsymb

        DFT_phn_tmp = []
        for nr in range(Nrf):
            T0 = tf.random.normal(shape=[N_symbols * self.K],
                              mean=0.0,
                              stddev=self.PHN_innovation_std,
                              dtype=tf.float32,
                              seed=None,
                              name=None)
            PHN_time = tf.math.cumsum(T0)
            PHN_time_reshaped = tf.reshape(PHN_time, shape=[N_symbols, self.K])
            exp_j_PHN_time_reshaped = tf.complex(tf.cos(PHN_time_reshaped),
                                                 tf.sin(PHN_time_reshaped)) / self.K
            DFT_of_exp_j_PHN_time_reshaped = tf.signal.fft(exp_j_PHN_time_reshaped)  # Computes the 1-dimensional discrete Fourier transform over the inner-most dimension of input
            DFT_phn_tmp.append(DFT_of_exp_j_PHN_time_reshaped)

        output = tf.transpose(tf.stack(DFT_phn_tmp, axis=0), perm=[1, 2, 0])

        return output

    
    def PHN_for_entire_batch(self, Nrf):
        DFT_of_exp_of_jPHN_tmp = []
        for ij in range(self.BATCHSIZE):
            DFT_of_exp_of_jPHN_tmp.append(self.Wiener_phase_noise_generator_Ruoyu_for_one_frame_forall_RF(Nrf))
        DFT_of_exp_of_jPHN = tf.stack(DFT_of_exp_of_jPHN_tmp, axis=0)
        return DFT_of_exp_of_jPHN

    
    def phase_noise_dataset_generator(self):
        # BS
        PHN_B_DFT_domain_samples_K_Nrf_train = self.PHN_for_entire_batch(self.N_b_rf)
        Lambda_B = self.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_B_DFT_domain_samples_K_Nrf_train)
        # UE
        PHN_U_DFT_domain_samples_K_Nrf_train = self.PHN_for_entire_batch(self.N_u_rf)
        Lambda_U = self.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_U_DFT_domain_samples_K_Nrf_train)
        return Lambda_B, Lambda_U

    # # the following phase noise is based on R. Zhang, B. Shim and H. Zhao, "Downlink Compressive Channel Estimation With Phase Noise in Massive MIMO Systems," in IEEE Transactions on Communications, vol. 68, no. 9, pp. 5534-5548, Sept. 2020, doi: 10.1109/TCOMM.2020.2998141.
    # def Wiener_phase_noise_generator_Ruoyu(self, N_rf):
    #     pn_std_sam = 2 * np.pi * self.fc * np.sqrt(self.c * self.Ts)
    #     N_frames = self.dataset_size
    #     PNsamps = np.float32(
    #         np.cumsum(np.random.normal(loc=0., scale=pn_std_sam, size=N_frames * self.Nsymb * self.K * N_rf)))
    #     PNsamps_cplx = (tf.complex(tf.cos(PNsamps), tf.sin(PNsamps))) / self.K
    #     PNsamps_cplx_K_Nrf = tf.reshape(PNsamps_cplx, shape=[N_frames, self.Nsymb, N_rf, self.K])
    #     DFT_PNsamps_cplx_K_Nrf = tf.signal.fft(PNsamps_cplx_K_Nrf)
    #     trans_DFT_PNsamps_cplx_K_Nrf = tf.transpose(DFT_PNsamps_cplx_K_Nrf, perm= [0, 1, 3, 2])  # batch, symb, k, rf
    #     return PNsamps_cplx_K_Nrf, trans_DFT_PNsamps_cplx_K_Nrf
    #
    # 
    # def phase_noise_dataset_generator(self):
    #     # BS
    #     dummy1, PHN_B_DFT_domain_samples_K_Nrf_train = self.Wiener_phase_noise_generator_Ruoyu(self.N_b_rf) #self.dataset_size * self.Nsymb * self.K * N_rf
    #     Lambda_B = self.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_B_DFT_domain_samples_K_Nrf_train)
    #     # UE
    #     dummy2, PHN_U_DFT_domain_samples_K_Nrf_train = self.Wiener_phase_noise_generator_Ruoyu(self.N_u_rf)
    #     Lambda_U = self.PHN_forall_RF_forall_K_forall_symbols_forall_samples(PHN_U_DFT_domain_samples_K_Nrf_train)
    #     return Lambda_B, Lambda_U

    
    def cyclical_shift(self, Lambda_matrix, k, flip):
        if flip == True:  # k-q
            return tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0)
        else:  # q-k
            return tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0)

    
    def non_zero_element_finder_for_H_tilde(self, k, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))),
                              self.K)  # zero indices for k-rolled fft sequence of phase noise
        ZI = tf.cast(ZI, dtype=tf.int64)
        s = ZI.shape
        mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                               values=tf.ones(shape=[s[0]],
                                                                                              dtype=tf.int32),
                                                                               dense_shape=[self.K]))
        mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)
        mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
                                                     shift=tf.squeeze(k) + 1, axis=0)
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(k), axis=0)
        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    
    def H_tilde_k_calculation(self, bundeled_inputs_0):
        H, Lambda_B, Lambda_U = bundeled_inputs_0
        T0 = tf.linalg.matmul(Lambda_U, H)
        T1 = tf.linalg.matmul(T0, Lambda_B)
        return T1

    
    def h_tilde_0_calculation_per_k(self, bundeled_inputs_0):  # inherits from paralle_loss_phase_noised_class
        H_forall_k, Lambda_B_0_forall_k, Lambda_U_0_forall_k, k = bundeled_inputs_0
        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        H_forall_k_masked = tf.boolean_mask(H_forall_k, mask=mask_of_ones, axis=0)
        Lambda_B_0_forall_k_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B_0_forall_k, k, flip=False),
                                                     mask=mask_of_ones, axis=0)
        Lambda_U_0_forall_k_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U_0_forall_k, k, flip=True),
                                                     mask=mask_of_ones, axis=0)
        bundeled_inputs_1 = [H_forall_k_masked, Lambda_B_0_forall_k_masked, Lambda_U_0_forall_k_masked]

        H_tilde_0_k = tf.cond(tf.equal(tf.size(H_forall_k_masked), 0),
                              lambda: tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype=tf.complex64),
                              lambda: tf.reduce_sum(tf.map_fn(self.H_tilde_k_calculation, bundeled_inputs_1,
                                                              fn_output_signature=tf.complex64,
                                                              parallel_iterations=int(
                                                                  self.K * self.truncation_ratio_keep)), axis=0))
        return H_tilde_0_k

    
    def h_tilde_0_calculation_forall_k(self, bundeled_inputs_0):
        H_forall_k, Lambda_B_0_forall_k, Lambda_U_0_forall_k = bundeled_inputs_0

        # repeating for function vectorization
        all_k = tf.reshape(tf.range(0, self.K, 1), shape=[self.K, 1])
        H_forall_k_repeated_K_times = tf.tile([H_forall_k], multiples=[self.K, 1, 1, 1])
        Lambda_B_0_forall_k_repeated_K_times = tf.tile([Lambda_B_0_forall_k], multiples=[self.K, 1, 1, 1])
        Lambda_U_0_forall_k_repeated_K_times = tf.tile([Lambda_U_0_forall_k], multiples=[self.K, 1, 1, 1])

        bundeled_inputs_1 = [H_forall_k_repeated_K_times, Lambda_B_0_forall_k_repeated_K_times,
                             Lambda_U_0_forall_k_repeated_K_times, all_k]
        H_tilde_0_forall_k = tf.map_fn(self.h_tilde_0_calculation_per_k, bundeled_inputs_1,
                                       fn_output_signature=tf.complex64,
                                       parallel_iterations=self.K)  # parallel over all k subcarriers
        return H_tilde_0_forall_k

    
    def h_tilde_0_calculation_forall_k_forall_samps(self,
                                                    bundeled_inputs_0):  # parallel over all samples of the dataset
        # H_forall_k_forall_samps, Lambda_B_0_forall_k_forall_samps, Lambda_U_0_forall_k_forall_samps = bundeled_inputs_0
        return tf.map_fn(self.h_tilde_0_calculation_forall_k, bundeled_inputs_0, fn_output_signature=tf.complex64)

    # CHANNEL LOADING AND GENERATION ///////////////////////////////////////////////////////////////////////////////////
    # sample_0 = { [H], [Lambda_0, Lambda_1,..., Lambda_13], [H_tilde_0 = f(H,Lambda_B_0, Lambda_U_0)]}
    # sample_1 = { [H], [Lambda_0, Lambda_1,..., Lambda_13], [H_tilde_0 = f(H,Lambda_B_0, Lambda_U_0)]}
    # so, Lambda_B/U have Nsymb times more samples

    
    def dataset_mapper(self, H_complex):
        Lambda_B, Lambda_U = self.phase_noise_dataset_generator()
        Lambda_B_0_forall_k_forall_samps = tf.squeeze(tf.slice(Lambda_B,
                                                               begin=[0, 0, 0, 0, 0],
                                                               size=[self.BATCHSIZE, 1, self.K, self.N_b_a,
                                                                     self.N_b_a]))
        Lambda_U_0_forall_k_forall_samps = tf.squeeze(tf.slice(Lambda_U,
                                                               begin=[0, 0, 0, 0, 0],
                                                               size=[self.BATCHSIZE, 1, self.K, self.N_u_a,
                                                                     self.N_u_a]))
        bundeled_inputs_0 = [H_complex, Lambda_B_0_forall_k_forall_samps, Lambda_U_0_forall_k_forall_samps]
        H_tilde_0_complex = self.h_tilde_0_calculation_forall_k_forall_samps(bundeled_inputs_0)
        H_tilde_0 = []
        H_tilde_0.append(tf.math.real(H_tilde_0_complex))
        H_tilde_0.append(tf.math.imag(H_tilde_0_complex))
        H_tilde_0 = tf.stack(H_tilde_0, axis=4)
        #
        # if (self.phase_noise == 'yes'):
        #     return H_complex, H_tilde_0, Lambda_B, Lambda_U
        # else:
        #     if (self.mode == 'train'):
        #         return H_tilde_0_complex, H_tilde_0
        #     else: # test and capacity evaluation
        #         return H_tilde_0_complex, H_tilde_0, H_complex, H_tilde_0, Lambda_B, Lambda_U

        return H_tilde_0_complex, H_tilde_0, H_complex, Lambda_B, Lambda_U

    
    def dataset_generator(self):
        mat_contents = sio.loadmat(self.mat_fname)
        H = np.zeros(shape=[self.dataset_size, self.K, self.N_u_a, self.N_b_a, 2], dtype=np.float32)
        if (self.mode == "train"):
            var_name_real = "H_real_" + "train"
            H[:, :, :, :, 0] = np.transpose(mat_contents[var_name_real], axes=[0, 3, 1, 2])[0:self.dataset_size, :, :,
                               :]
            var_name_imag = "H_imag_" + "train"
            H[:, :, :, :, 1] = np.transpose(mat_contents[var_name_imag], axes=[0, 3, 1, 2])[0:self.dataset_size, :, :,
                               :]
            H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        else:
            var_name_real = "H_real_" + "test"
            H[:, :, :, :, 0] = \
                np.transpose(mat_contents[var_name_real], axes=[0, 3, 1, 2])[0:self.dataset_size, :, :, :]
            var_name_imag = "H_imag_" + "test"
            H[:, :, :, :, 1] = \
                np.transpose(mat_contents[var_name_imag], axes=[0, 3, 1, 2])[0:self.dataset_size, :, :, :]

        H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        my_dataset = tf.data.Dataset.from_tensor_slices(H_complex)
        my_dataset = my_dataset.cache()
        my_dataset = my_dataset.batch(self.BATCHSIZE)
        my_dataset = my_dataset.map(self.dataset_mapper, num_parallel_calls=tf.data.AUTOTUNE)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        my_dataset = my_dataset.prefetch(AUTOTUNE)
        return my_dataset

    
    def data_generator_for_evaluation_of_proposed_beamformer(self, batch_number):
        mat_contents = sio.loadmat(self.mat_fname)
        H = np.zeros(shape=[self.BATCHSIZE, self.K, self.N_u_a, self.N_b_a, 2], dtype=np.float32)
        var_name_real = "H_real_" + "test"
        H[:, :, :, :, 0] = \
            np.transpose(mat_contents[var_name_real], axes=[0, 3, 1, 2])[
            batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]
        var_name_imag = "H_imag_" + "test"
        H[:, :, :, :, 1] = \
            np.transpose(mat_contents[var_name_imag], axes=[0, 3, 1, 2])[
            batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]

        H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        Lambda_B, Lambda_U = self.phase_noise_dataset_generator()
        Lambda_B_0_forall_k_forall_samps = tf.squeeze(tf.slice(Lambda_B,
                                                               begin=[0, 0, 0, 0, 0],
                                                               size=[self.BATCHSIZE, 1, self.K, self.N_b_a,
                                                                     self.N_b_a]))
        Lambda_U_0_forall_k_forall_samps = tf.squeeze(tf.slice(Lambda_U,
                                                               begin=[0, 0, 0, 0, 0],
                                                               size=[self.BATCHSIZE, 1, self.K, self.N_u_a,
                                                                     self.N_u_a]))

        bundeled_inputs_0 = [H_complex, Lambda_B_0_forall_k_forall_samps, Lambda_U_0_forall_k_forall_samps]
        H_tilde_0_complex = self.h_tilde_0_calculation_forall_k_forall_samps(bundeled_inputs_0)
        H_tilde_0 = []
        H_tilde_0.append(tf.math.real(H_tilde_0_complex))
        H_tilde_0.append(tf.math.imag(H_tilde_0_complex))
        H_tilde_0 = tf.stack(H_tilde_0, axis=4)
        return H_complex, H_tilde_0, Lambda_B, Lambda_U

    
    def data_generator_for_evaluation_of_Sohrabis_beamformer(self, batch_number):
        mat_contents = sio.loadmat(self.mat_fname)
        # No permutation is needed for the following data because they are not modified in Matlab and merely were passed
        # to the matlab code and came back here without any changes (so their sizes is also Ok)
        H_complex = (mat_contents['H'])[batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]
        # H_tilde_0 = (mat_contents['H_tilde_0'])[batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]
        Lambda_B = (mat_contents['Lambda_B'])[batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :,
                   :, :]
        Lambda_U = (mat_contents['Lambda_U'])[batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :,
                   :, :]

        V_RF_Sohrabi_optimized = (mat_contents['V_RF_Sohrabi_optimized'])[
                                 batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :]
        W_RF_Sohrabi_optimized = (mat_contents['W_RF_Sohrabi_optimized'])[
                                 batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :]

        # The following data require permutation to bring k (subcarrier) to the second dimension
        V_D_Sohrabi_optimized = np.transpose(mat_contents['V_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]
        W_D_Sohrabi_optimized = np.transpose(mat_contents['W_D_Sohrabi_optimized'], axes=[0, 3, 1, 2])[
                                batch_number * self.BATCHSIZE: (batch_number + 1) * self.BATCHSIZE, :, :, :]
        return H_complex, Lambda_B, Lambda_U, V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, \
               V_D_Sohrabi_optimized, W_D_Sohrabi_optimized
