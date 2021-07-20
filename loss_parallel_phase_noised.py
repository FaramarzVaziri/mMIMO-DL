import numpy as np
import tensorflow as tf


class paralle_loss_phase_noised_class:

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

    @tf.function
    def cyclical_shift(self, Lambda_matrix, k, flip):
        if flip == True:  # k-q
            return tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0)
        else:  # q-k
            return tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0)

    @tf.function
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
    #
    # @tf.function
    # def non_zero_element_finder_for_H_tilde_ft(self, k, truncation_ratio_keep):  # flip true
    #     z = 1 - truncation_ratio_keep
    #     B_orig = int(
    #         self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
    #     ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))),
    #                           self.K)  # zero indices for k-rolled fft sequence of phase noise
    #     ZI = tf.cast(ZI, dtype=tf.int64)
    #     s = ZI.shape
    #     mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
    #                                                                            values=tf.ones(shape=[s[0]],
    #                                                                                           dtype=tf.int32),
    #                                                                            dense_shape=[self.K]))
    #     mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)
    #     mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
    #                                                  shift=tf.squeeze(k) + 1, axis=0)
    #     return mask_of_ones_after_shift_flip_true
    #
    # @tf.function
    # def non_zero_element_finder_for_H_tilde_ff(self, k, truncation_ratio_keep):  # flip false
    #     z = 1 - truncation_ratio_keep
    #     B_orig = int(
    #         self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
    #     ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))),
    #                           self.K)  # zero indices for k-rolled fft sequence of phase noise
    #     ZI = tf.cast(ZI, dtype=tf.int64)
    #     s = ZI.shape
    #     mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
    #                                                                            values=tf.ones(shape=[s[0]],
    #                                                                                           dtype=tf.int32),
    #                                                                            dense_shape=[self.K]))
    #     mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)
    #     mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(k), axis=0)
    #     return mask_of_ones_after_shift_flip_false

    # R_X calculations /////////////////////////////////////////////////////////////////////////////////////////////////

    @tf.function
    def H_tilde_k_calculation(self, bundeled_inputs_0):
        H_k, Lambda_B_k, Lambda_U_k = bundeled_inputs_0
        T0 = tf.linalg.matmul(Lambda_U_k, H_k)
        T1 = tf.linalg.matmul(T0, Lambda_B_k)
        return T1

    @tf.function
    def Rx_calculation_per_k(self, bundeled_inputs_0):
        V_D_k, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_B_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, k, flip=False),
                                          mask=mask_of_ones, axis=0)
        Lambda_U_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, k, flip=True),
                                          mask=mask_of_ones, axis=0)
        bundeled_inputs_1 = [H_masked, Lambda_B_masked, Lambda_U_masked]

        H_tilde_k = tf.cond(tf.equal(tf.size(H_masked), 0),
                            lambda: tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype=tf.complex64),
                            lambda: tf.reduce_sum(tf.map_fn(self.H_tilde_k_calculation, bundeled_inputs_1,
                                                            fn_output_signature=tf.complex64, parallel_iterations=int(
                                    self.K * self.truncation_ratio_keep)), axis=0))

        T1 = tf.linalg.matmul(T0, H_tilde_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        A_ns_k = tf.linalg.matmul(T2, V_D_k)
        R_X_k = tf.linalg.matmul(A_ns_k, A_ns_k, adjoint_a=False, adjoint_b=True)
        return R_X_k

    @tf.function
    def Rx_calculation_forall_k(self, bundeled_inputs_0):
        V_D_forsome_k, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times, W_RF_repeated_K_times, Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K = bundeled_inputs_0
        #
        # Lambda_B_forall_k_repeated_K_times = tf.tile([Lambda_B_forall_k], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        # Lambda_U_forall_k_repeated_K_times = tf.tile([Lambda_U_forall_k], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])

        bundeled_inputs_1 = [V_D_forsome_k, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times,
                             W_RF_repeated_K_times,
                             Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K]
        R_X = tf.map_fn(self.Rx_calculation_per_k, bundeled_inputs_1, fn_output_signature=tf.complex64,
                        parallel_iterations=int(
                            self.sampling_ratio_subcarrier_domain_keep * self.K))  # parallel over all k subcarriers
        return R_X

    # R_Q calculations /////////////////////////////////////////////////////////////////////////////////////////////////

    @tf.function
    def non_zero_element_finder_for_H_hat(self, k, m, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int( self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + tf.range(int(self.K * z)), self.K)  # zero indices for k-rolled fft sequence of phase noise
        # ZI = tf.math.floormod(B_orig + np.array(range(int(self.K * z))), self.K)  # zero indices for k-rolled fft sequence of phase noise
        ZI = tf.cast(ZI, dtype=tf.int64)
        s = ZI.shape
        mask_of_zeros_before_shift = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=tf.reshape(ZI, shape=[s[0], 1]),
                                                                               values=tf.ones(shape=[s[0]], dtype=tf.int32),
                                                                               dense_shape=[self.K]))
        mask_of_ones_before_shift = tf.subtract(1, mask_of_zeros_before_shift)

        mask_of_ones_after_shift_flip_true = tf.roll(tf.reverse(mask_of_ones_before_shift, axis=[0]),
                                                     shift=tf.squeeze(k) + 1, axis=0)
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(m), axis=0)

        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    @tf.function
    def H_hat_m_k_calculation(self, bundeled_inputs):
        H_k, Lambda_B_k, Lambda_U_k = bundeled_inputs
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_k, H_k), Lambda_B_k)

    @tf.function
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

    @tf.function
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

    @tf.function
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
        all_m = tf.range(0, self.K, 1, dtype= tf.int32)
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

    @tf.function
    def Rq_calculation_forall_k(self, bundeled_inputs_0):
        # V_D_repeated_K_times, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times, W_RF_repeated_K_times, \
        # Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K = bundeled_inputs_0
        #
        # bundeled_inputs_1 = [V_D_repeated_K_times, W_D_forsome_k, H_repeated_K_times,
        #                      V_RF_repeated_K_times, W_RF_repeated_K_times, Lambda_B_repeated_K_times,
        #                      Lambda_U_repeated_K_times, sampled_K]
        R_Q = tf.map_fn(self.Rq_calculation_per_k, bundeled_inputs_0, fn_output_signature=tf.complex64,
                        parallel_iterations=int( self.sampling_ratio_subcarrier_domain_keep * self.K))  # parallel over all K subcarriers
        return R_Q

    # Capacity calculation
    @tf.function
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

    @tf.function
    def capacity_calculation_forall_k(self, bundeled_inputs_0):  # K*...
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # one sample of batch, RFs are not forall k

        sampled_K = tf.convert_to_tensor(
            np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False), dtype=tf.int64)
        # print(sampled_K.shape)

        sampled_K = tf.reshape(sampled_K, shape=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1])
        mask_of_subcarriers = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(indices=sampled_K,
                                      values=tf.ones(shape=[int(self.sampling_ratio_subcarrier_domain_keep * self.K)], dtype=tf.int32),
                                      dense_shape=[self.K])))
        sampled_K = tf.cast(sampled_K, dtype= tf.int32)
        W_D_forsome_k = tf.boolean_mask(W_D, mask=mask_of_subcarriers, axis=0)
        V_D_forsome_k = tf.boolean_mask(V_D, mask=mask_of_subcarriers, axis=0)

        W_RF_repeated_K_times = tf.tile([W_RF], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
        H_repeated_K_times = tf.tile([H], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        Lambda_B_repeated_K_times = tf.tile([Lambda_B], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        Lambda_U_repeated_K_times = tf.tile([Lambda_U], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        V_RF_repeated_K_times = tf.tile([V_RF], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
        V_D_repeated_K_times = tf.tile([V_D], multiples=[int(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])

        bundeled_inputs_1 = [V_D_forsome_k, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times,
                             W_RF_repeated_K_times, Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K]
        RX_forall_k = self.Rx_calculation_forall_k(bundeled_inputs_1)

        bundeled_inputs_2 = [V_D_repeated_K_times, W_D_forsome_k, H_repeated_K_times, V_RF_repeated_K_times,
                             W_RF_repeated_K_times, Lambda_B_repeated_K_times, Lambda_U_repeated_K_times, sampled_K]

        RQ_forall_k = self.Rq_calculation_forall_k(bundeled_inputs_2)
        bundeled_inputs2 = [RX_forall_k, RQ_forall_k]
        C = tf.reduce_mean(tf.map_fn(self.capacity_calculation_per_k,
                                     bundeled_inputs2,
                                     fn_output_signature=tf.float32,
                                     parallel_iterations=int(self.sampling_ratio_subcarrier_domain_keep * self.K)), axis=0)
        return C, RX_forall_k, RQ_forall_k

    @tf.function
    def capacity_calculation_for_frame(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        # repeating inputs for vectorization
        V_D_repeated_Nsymb_times = tf.tile([V_D],
                                           multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1,  1, 1])
        W_D_repeated_Nsymb_times = tf.tile([W_D],
                                           multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1, 1])
        H_repeated_Nsymb_times = tf.tile([H],
                                         multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1, 1])
        V_RF_repeated_Nsymb_times = tf.tile([V_RF],
                                            multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1])
        W_RF_repeated_Nsymb_times = tf.tile([W_RF],
                                            multiples=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1, 1])

        selected_symbols = tf.convert_to_tensor(
            np.random.choice(self.Nsymb, int(self.sampling_ratio_time_domain_keep * self.Nsymb), replace=False), dtype=tf.int64)
        # print(selected_symbols.shape)

        sampled_Nsymb = tf.reshape(selected_symbols,
                                   shape=[int(self.sampling_ratio_time_domain_keep * self.Nsymb), 1])

        mask_of_symbols = tf.sparse.to_dense(
            tf.sparse.reorder(
                tf.sparse.SparseTensor(
                    indices= sampled_Nsymb,
                    values=tf.ones(shape=[int(self.sampling_ratio_time_domain_keep * self.Nsymb)], dtype=tf.int32),
                    dense_shape=[self.Nsymb])))
        sampled_Nsymb = tf.cast(sampled_Nsymb, dtype = tf.int32)
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

    @tf.function
    def capacity_calculation_for_frame_for_batch(self, bundeled_inputs_0):
        capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples = \
            tf.map_fn(self.capacity_calculation_for_frame, bundeled_inputs_0,
                      fn_output_signature=(tf.float32, tf.complex64, tf.complex64), parallel_iterations=self.BATCHSIZE)

        return tf.multiply(-1.0, tf.reduce_mean(tf.reduce_mean(capacity_sequence_in_frame_forall_samples, axis=0), axis=1)), \
               capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples