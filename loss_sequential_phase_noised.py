import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

class sequential_loss_phase_noised_class:

    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb,
                 sampling_ratio_time_domain_keep, sampling_ratio_subcarrier_domain_keep):
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
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.sampling_ratio_subcarrier_domain_keep = sampling_ratio_subcarrier_domain_keep


    def cyclical_shift(self, Lambda_matrix, k, flip):
        if flip == True:  # k-q
            return tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0)
        else:  # q-k
            return tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0)

    def non_zero_element_finder_for_H_tilde(self, k, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + tf.range(int(self.K * z)), self.K)  # zero indices for k-rolled fft sequence of phase noise
        # ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))), self.K)  # zero indices for k-rolled fft sequence of phase noise
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
        H_k, Lambda_B_k, Lambda_U_k = bundeled_inputs_0

        # T1 = tnp.matmul(tnp.matmul(Lambda_U_k, H_k), Lambda_B_k)
        T1 = tf.linalg.matmul(tf.linalg.matmul(Lambda_U_k, H_k), Lambda_B_k)
        return T1

    def Rx_calculation_per_k(self, bundeled_inputs_0):
        V_D_k, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_B_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, k, flip=False),
                                          mask=mask_of_ones, axis=0)
        Lambda_U_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, k, flip=True),
                                          mask=mask_of_ones, axis=0)

        # # todo- speed comment: Changing map_fn to for loop made it 5.8 times faster than mpa_fn
        # H_tilde_k = 0
        # for k in range(int(self.K * self.truncation_ratio_keep)):
        #     H_tilde_k = H_tilde_k + self.H_tilde_k_calculation([H_masked[k,:], Lambda_B_masked[k,:], Lambda_U_masked[k,:]])

        # # todo- speed comment: Changing python slicing to tf.slice made it 7 times faster than map_fn
        # H_tilde_k = 0
        # for k in range(int(self.K * self.truncation_ratio_keep)):
        #     H_tilde_k = H_tilde_k + self.H_tilde_k_calculation([tf.slice(H_masked, [k, 0, 0], [1, self.N_u_a, self.N_b_a]), tf.slice(Lambda_B_masked, [k, 0, 0], [1, self.N_b_a, self.N_b_a]), tf.slice(Lambda_U_masked, [k, 0, 0], [1, self.N_u_a, self.N_u_a])])
        #

        bundeled_inputs_1 = [H_masked, Lambda_B_masked, Lambda_U_masked]
        H_tilde_k = tf.reduce_sum(tf.map_fn(self.H_tilde_k_calculation, bundeled_inputs_1, fn_output_signature=tf.complex64, parallel_iterations=int(self.K * self.truncation_ratio_keep)), axis=0)

        T1 = tf.linalg.matmul(T0, H_tilde_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        A_ns_k = tf.linalg.matmul(T2, V_D_k)
        R_X_k = tf.linalg.matmul(A_ns_k, A_ns_k, adjoint_a=False, adjoint_b=True)
        return R_X_k

    def Rx_calculation_forall_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, sampled_K = bundeled_inputs_0
        R_X_tmp = []#[None] * int(self.sampling_ratio_subcarrier_domain_keep * self.K)
        for k in sampled_K:
            R_X_tmp.append(self.Rx_calculation_per_k([V_D[k, :], W_D[k, :], H, V_RF, W_RF, Lambda_B, Lambda_U, k]))
        R_X = tf.stack(R_X_tmp)
        return R_X

    
    def non_zero_element_finder_for_H_hat(self, k, m, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + tf.range(int(self.K * z)),
                              self.K)  # zero indices for k-rolled fft sequence of phase noise
        # ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))), self.K)  # zero indices for k-rolled fft sequence of phase noise
        ZI = tf.cast(ZI, dtype=tf.int64)
        s = ZI.shape
        mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                               values=tf.ones(shape=[s[0]],
                                                                                              dtype=tf.int32),
                                                                               dense_shape=[self.K]))
        mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)

        mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
                                                     shift=tf.squeeze(k) + 1, axis=0)
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(m), axis=0)

        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    def H_hat_m_k_calculation(self, bundeled_inputs):
        H_k, Lambda_B_k, Lambda_U_k = bundeled_inputs
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_k, H_k), Lambda_B_k)

    def R_I_Q_m_k(self, bundeled_inputs_0):
        H, Lambda_B, Lambda_U, V_D, V_RF, W_D, W_RF, k, m = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D, W_RF, adjoint_a=True, adjoint_b=True)
        mask_of_ones = self.non_zero_element_finder_for_H_hat(tf.squeeze(k), tf.squeeze(m), self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_B_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, tf.squeeze(m), flip=False),
                                          mask=mask_of_ones, axis=0)
        Lambda_U_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, tf.squeeze(k), flip=True),
                                          mask=mask_of_ones, axis=0)
        bundeled_inputs_1 = [H_masked, Lambda_B_masked, Lambda_U_masked]
        H_hat_m_k = tf.cond(tf.equal(tf.size(H_masked), 0),
                            lambda: tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype=tf.complex64),
                            lambda: tf.reduce_sum(tf.map_fn(self.H_hat_m_k_calculation, bundeled_inputs_1,
                                                            fn_output_signature=tf.complex64, parallel_iterations=int(
                                    self.K * self.truncation_ratio_keep)), axis=0))

        T1 = tf.linalg.matmul(T0, H_hat_m_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        B_m_k = tf.linalg.matmul(T2, V_D)
        R = tf.linalg.matmul(B_m_k, B_m_k, adjoint_a=False, adjoint_b=True)
        return R

    def R_N_Q_m_k(self, bundeled_inputs_0):
        Lambda_U, W_D, W_RF, k, m = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D, W_RF, adjoint_a=True, adjoint_b=True)
        # size_Lambda_U = Lambda_U.shape
        Lambda_U_for_k_subtract_m = tf.reshape(tf.slice(Lambda_U,
                                                        begin=[tf.math.floormod(tf.squeeze(k) - tf.squeeze(m), self.K),
                                                               0, 0],
                                                        size=[1, self.N_u_a, self.N_u_a]),
                                               shape=[self.N_u_a, self.N_u_a])

        C_m_k = tf.linalg.matmul(T0, Lambda_U_for_k_subtract_m)
        R = self.sigma2 * tf.linalg.matmul(C_m_k, C_m_k, adjoint_a=False, adjoint_b=True)
        return R

    
    def Rq_calculation_per_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0

        # repeating inputs for the vectorized loop over m-1 subcarriers for R_I_Q_m_k __________________________________
        all_m_except_k = tf.concat([tf.reshape(tf.range(0, tf.squeeze(k), 1), shape=[tf.squeeze(k), 1]),
                                    tf.reshape(tf.range(tf.squeeze(k) + 1, self.K, 1),
                                               shape=[self.K - tf.squeeze(k) - 1, 1])], axis=0)
        H_repeated_K_1_times = tf.tile([H], multiples=[self.K - 1, 1, 1, 1])
        Lambda_B_repeated_K_1_times = tf.tile([Lambda_B], multiples=[self.K - 1, 1, 1, 1])
        Lambda_U_repeated_K_1_times = tf.tile([Lambda_U], multiples=[self.K - 1, 1, 1, 1])
        ZI = tf.cast(k, dtype=tf.int64)
        s = ZI.shape
        mask_of_zero = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                 values=tf.ones(shape=[s[0]], dtype=tf.int32),
                                                                 dense_shape=[self.K]))
        mask_of_one = tf.subtract(1, mask_of_zero)
        V_D_except_k = tf.boolean_mask(V_D, mask=mask_of_one, axis=0)
        W_D_repeated_K_1_times = tf.tile([W_D], multiples=[self.K - 1, 1, 1])
        V_RF_repeated_K_1_times = tf.tile([V_RF], multiples=[self.K - 1, 1, 1])
        W_RF_repeated_K_1_times = tf.tile([W_RF], multiples=[self.K - 1, 1, 1])
        k_repeated_K_1_times = tf.tile([k], multiples=[self.K - 1, 1])
        # ______________________________________________________________________________________________________________

        bundeled_inputs_1 = [H_repeated_K_1_times, Lambda_B_repeated_K_1_times,
                             Lambda_U_repeated_K_1_times, V_D_except_k, V_RF_repeated_K_1_times,
                             W_D_repeated_K_1_times, W_RF_repeated_K_1_times, k_repeated_K_1_times, all_m_except_k]

        # repeating inputs for the vectorized loop over m-1 subcarriers for R_N_Q_m_k __________________________________
        all_m = tf.range(0, self.K, 1, dtype=tf.int32)
        # all_m = tf.range(0, self.K, 1)
        Lambda_U_repeated_K_times = tf.tile([Lambda_U], multiples=[self.K, 1, 1, 1])
        W_D_repeated_K_times = tf.tile([W_D], multiples=[self.K, 1, 1])
        W_RF_repeated_K_times = tf.tile([W_RF], multiples=[self.K, 1, 1])
        k_repeated_K_times = tf.tile([k], multiples=[self.K, 1])
        # ______________________________________________________________________________________________________________

        bundeled_inputs_2 = [Lambda_U_repeated_K_times, W_D_repeated_K_times, W_RF_repeated_K_times,
                             k_repeated_K_times, all_m]

        R_Q = tf.add(tf.reduce_sum(tf.map_fn(self.R_I_Q_m_k, bundeled_inputs_1, fn_output_signature=tf.complex64,
                                             parallel_iterations=self.K - 1), axis=0),
                     tf.reduce_sum(tf.map_fn(self.R_N_Q_m_k, bundeled_inputs_2, fn_output_signature=tf.complex64,
                                             parallel_iterations=self.K), axis=0))
        return R_Q

    def Rq_calculation_forall_k(self, bundeled_inputs_0):
        # V_D_repeated_K_times, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times, W_RF_repeated_K_times, \
        # Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K = bundeled_inputs_0
        #
        # bundeled_inputs_1 = [V_D_repeated_K_times, W_D_forsome_k, H_repeated_K_times,
        #                      V_RF_repeated_K_times, W_RF_repeated_K_times, Lambda_B_repeated_K_times,
        #                      Lambda_U_repeated_K_times, sampled_K]
        R_Q = tf.map_fn(self.Rq_calculation_per_k, bundeled_inputs_0, fn_output_signature=tf.complex64,
                        parallel_iterations=int(
                            self.sampling_ratio_subcarrier_domain_keep * self.K))  # parallel over all K subcarriers
        return R_Q

    # Capacity calculation

    def capacity_calculation_per_k(self, bundeled_inputs_0):
        R_X, R_Q = bundeled_inputs_0
        precision_fixer = 1e-7
        R_Q = tf.add(precision_fixer * tf.eye(self.N_s, dtype=tf.complex64), R_Q)  # numeric precision flaw
        T0 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.complex64), tf.linalg.det(R_Q)),
                     lambda: tf.multiply(tf.zeros([1], dtype=tf.complex64), R_Q),
                     lambda: tf.linalg.inv(R_Q))
        T1 = tf.linalg.matmul(T0, R_X, adjoint_a=False, adjoint_b=False)
        T2 = tf.add(tf.eye(self.N_s, dtype=tf.complex64), T1)
        T3 = tf.math.real(tf.linalg.det(T2))
        eta = 0.
        return tf.cond(tf.less(0.0, T3),
                       lambda: tf.divide(tf.math.log(T3), tf.math.log(2.0)),
                       lambda: tf.multiply(eta, T3))

    def capacity_calculation_forall_k(self, bundeled_inputs_0):  # K*...
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # one sample of batch, RFs are not forall k

        sampled_K = np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False)

        bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, sampled_K]
        RX_forall_k = self.Rx_calculation_forall_k(bundeled_inputs_1)

        RQ_forall_k = self.Rq_calculation_forall_k(bundeled_inputs_1)
        bundeled_inputs2 = [RX_forall_k, RQ_forall_k]
        C = tf.reduce_mean(tf.map_fn(self.capacity_calculation_per_k,
                                     bundeled_inputs2,
                                     fn_output_signature=tf.float32,
                                     parallel_iterations=int(self.sampling_ratio_subcarrier_domain_keep * self.K)),
                           axis=0)
        return C, RX_forall_k, RQ_forall_k

    def capacity_calculation_for_frame(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        # repeating inputs for vectorization
        V_D_repeated_Nsymb_times = tf.tile([V_D],
                                           multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1, 1])
        W_D_repeated_Nsymb_times = tf.tile([W_D],
                                           multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1, 1])
        H_repeated_Nsymb_times = tf.tile([H],
                                         multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1, 1])
        V_RF_repeated_Nsymb_times = tf.tile([V_RF],
                                            multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1])
        W_RF_repeated_Nsymb_times = tf.tile([W_RF],
                                            multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1])

        selected_symbols = tf.convert_to_tensor(
            np.random.choice(self.Nsymb, int(self.sampling_ratio_time_domain_keep * self.Nsymb), replace=False),
            dtype=tf.int64)
        # print(selected_symbols.shape)

        sampled_Nsymb = tf.reshape(selected_symbols,
                                   shape=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1])

        mask_of_symbols = tf.sparse.to_dense(
            tf.sparse.reorder(
                tf.sparse.SparseTensor(
                    indices=sampled_Nsymb,
                    values=tf.ones(shape=[int(self.sampling_ratio_time_domain_keep * self.Nsymb)], dtype=tf.int32),
                    dense_shape=[self.Nsymb])))
        sampled_Nsymb = tf.cast(sampled_Nsymb, dtype=tf.int32)
        Lambda_B_sampled = tf.boolean_mask(Lambda_B, mask=mask_of_symbols, axis=0)
        Lambda_U_sampled = tf.boolean_mask(Lambda_U, mask=mask_of_symbols, axis=0)

        bundeled_inputs_1 = [V_D_repeated_Nsymb_times, W_D_repeated_Nsymb_times, H_repeated_Nsymb_times,
                             V_RF_repeated_Nsymb_times, W_RF_repeated_Nsymb_times, Lambda_B_sampled, Lambda_U_sampled]

        ergodic_capacity_forall_OFDMs, RX_forall_k_forall_OFDMs, RQ_forall_k_forall_OFDMs = tf.map_fn(
            self.capacity_calculation_forall_k, bundeled_inputs_1,
            fn_output_signature=(tf.float32, tf.complex64, tf.complex64),
            parallel_iterations=int(self.sampling_ratio_time_domain_keep * self.Nsymb))

        # print(ergodic_capacity_forall_OFDMs.numpy())
        # print(';')
        capacity_sequence_in_frame = tf.reshape(ergodic_capacity_forall_OFDMs,
                                                shape=[1, int(self.sampling_ratio_time_domain_keep * self.Nsymb)])
        # print(capacity_sequence_in_frame.shape)
        # C_average_of_frame = tf.reduce_mean(ergodic_capacity_forall_OFDMs, axis=0)
        # print(C_average_of_frame.shape)
        return capacity_sequence_in_frame, RX_forall_k_forall_OFDMs, RQ_forall_k_forall_OFDMs

    def capacity_calculation_for_frame_for_batch(self, bundeled_inputs_0):
        # impl with map_fn
        capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples = \
            tf.map_fn(self.capacity_calculation_for_frame, bundeled_inputs_0,
                      fn_output_signature=(tf.float32, tf.complex64, tf.complex64), parallel_iterations=self.BATCHSIZE)

        return tf.multiply(-1.0,
                           tf.reduce_mean(tf.reduce_mean(capacity_sequence_in_frame_forall_samples, axis=0), axis=1)), \
               capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples

        # # impl with for ------------------------------------------------------------------------------------------------
        # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        # capacity_sequence_in_frame_forall_samples_tmp = []
        # RX_forall_k_forall_OFDMs_forall_samples_tmp = []
        # RQ_forall_k_forall_OFDMs_forall_samples_tmp = []
        # for ij in range(self.BATCHSIZE):
        #     T = self.capacity_calculation_for_frame([V_D[ij,:], W_D[ij,:], H[ij,:], V_RF[ij,:], W_RF[ij,:], Lambda_B[ij,:], Lambda_U[ij,:]])
        #     capacity_sequence_in_frame_forall_samples_tmp.append(T[0])
        #     RX_forall_k_forall_OFDMs_forall_samples_tmp.append(T[1])
        #     RQ_forall_k_forall_OFDMs_forall_samples_tmp.append(T[2])
        #
        # capacity_sequence_in_frame_forall_samples = tf.stack(capacity_sequence_in_frame_forall_samples_tmp, axis=0)
        # RX_forall_k_forall_OFDMs_forall_samples = tf.stack(RX_forall_k_forall_OFDMs_forall_samples_tmp, axis=0)
        # RQ_forall_k_forall_OFDMs_forall_samples = tf.stack(RQ_forall_k_forall_OFDMs_forall_samples_tmp, axis=0)
        #
        # return -1.0*tf.reduce_mean(capacity_sequence_in_frame_forall_samples, axis=0), \
        #        capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples
