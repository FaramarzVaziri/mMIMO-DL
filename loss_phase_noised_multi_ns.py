import numpy as np
import tensorflow as tf


class loss_phase_noised_class:

    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, truncation_ratio_keep, Nsymb,
                 sampling_ratio_time_domain_keep, sampling_ratio_subcarrier_domain_keep, mode, impl):
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
        self.truncation_ratio_keep = truncation_ratio_keep
        self.Nsymb = Nsymb
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.sampling_ratio_subcarrier_domain_keep = sampling_ratio_subcarrier_domain_keep
        self.mode = mode
        self.impl = impl

    @tf.function
    def cyclical_shift(self, Lambda_matrix, k, flip):
        if flip == True:  # k-q
            return tf.roll(tf.reverse(Lambda_matrix, axis=[0]), shift=tf.squeeze(k) + 1, axis=0)
        else:  # q-k
            return tf.roll(Lambda_matrix, shift=tf.squeeze(k), axis=0)

    @tf.function
    def non_zero_element_finder_for_H_tilde(self, k, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        # print('tf.range(int(self.K * z)): ', tf.range(int(self.K * z)))
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
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(k), axis=0)
        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total


    @tf.function
    def Rx_per_k(self, bundeled_inputs_0):
        V_D_k, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0  # [k, ...]
        # todo: non-zero element finder bypassed
        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        # mask_of_ones = tf.range(self.K)
        if (tf.reduce_sum(mask_of_ones) == 0):
            RX_k = tf.zeros(shape=[self.N_s, self.N_s], dtype=tf.float32)
        else:
            # print('k =', k, ' mask_of_ones = ', mask_of_ones)
            H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
            Lambda_U_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, k, flip=True),
                                                          mask=mask_of_ones,
                                                          axis=0)
            Lambda_B_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, k, flip=False),
                                                          mask=mask_of_ones,
                                                          axis=0)
            # H_tilde_k = tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype=tf.complex64)
            # for q in range(round(self.K * self.truncation_ratio_keep)):
            #     H_tilde_k = tf.add(H_tilde_k,
            #                        tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked[q],
            #                                                          H_masked[q]), Lambda_B_cyclshifted_masked[q]))
            bundeled_inputs_1 = [H_masked, Lambda_B_cyclshifted_masked, Lambda_U_cyclshifted_masked]
            H_tilde_k = tf.reduce_sum(tf.map_fn(self.H_tilde_k_q, bundeled_inputs_1, fn_output_signature=tf.complex64),
                                      axis=0)

            T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
            T1 = tf.linalg.matmul(T0, H_tilde_k)
            T2 = tf.linalg.matmul(T1, V_RF)
            A_ns_k = tf.linalg.matmul(T2, V_D_k)
            RX_k = tf.cast(tf.linalg.matmul(A_ns_k, A_ns_k, adjoint_a=False, adjoint_b=True), dtype=tf.float32)
        return RX_k

    @tf.function
    def H_tilde_k_q(self, bundeled_inputs_0):
        H_masked, Lambda_B_cyclshifted_masked, Lambda_U_cyclshifted_masked = bundeled_inputs_0
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked,
                                                 H_masked),
                                Lambda_B_cyclshifted_masked)

    @tf.function
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

    @tf.function
    def R_I_Q_m_k(self, bundeled_inputs_0):
        V_D_m, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k, m = bundeled_inputs_0
        # todo: non-zero element finder bypassed
        mask_of_ones = self.non_zero_element_finder_for_H_hat(k, m, self.truncation_ratio_keep)
        # mask_of_ones = tf.range(self.K)
        if ((m == k) or (tf.reduce_sum(mask_of_ones) == 0)):
            R = tf.zeros(shape=[self.N_s, self.N_s], dtype=tf.complex64)
        else:
            # print('m =', m, ' mask_of_ones = ', mask_of_ones)
            H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
            Lambda_B_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, tf.squeeze(m), flip=False),
                                                          mask=mask_of_ones, axis=0)
            Lambda_U_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, tf.squeeze(k), flip=True),
                                                          mask=mask_of_ones, axis=0)
            # H_hat_m_k = tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype=tf.complex64)
            # for q in tf.range(round(self.K * self.truncation_ratio_keep)):
            #     H_hat_m_k = tf.add(H_hat_m_k, tf.linalg.matmul(
            #         tf.linalg.matmul(Lambda_U_cyclshifted_masked[q, :], H_masked[q, :], adjoint_a=False,
            #                          adjoint_b=False),
            #         Lambda_B_cyclshifted_masked[q, :], adjoint_a=False, adjoint_b=False))
            bundeled_inputs_1 = [H_masked, Lambda_B_cyclshifted_masked, Lambda_U_cyclshifted_masked]
            H_hat_m_k = tf.reduce_sum(tf.map_fn(self.H_hat_m_k_q, bundeled_inputs_1, fn_output_signature=tf.complex64),
                                      axis=0)
            T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
            T1 = tf.linalg.matmul(T0, H_hat_m_k)
            T2 = tf.linalg.matmul(T1, V_RF)
            B_m_k = tf.linalg.matmul(T2, V_D_m)
            R = tf.linalg.matmul(B_m_k, B_m_k, adjoint_a=False, adjoint_b=True)
        return R

    @tf.function
    def H_hat_m_k_q(self, bundeled_inputs_0):
        H_masked, Lambda_B_cyclshifted_masked, Lambda_U_cyclshifted_masked = bundeled_inputs_0
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked,
                                                 H_masked, adjoint_a=False, adjoint_b=False),
                                Lambda_B_cyclshifted_masked, adjoint_a=False, adjoint_b=False)

    @tf.function
    def R_N_Q_m_k(self, bundeled_inputs_0):
        Lambda_U_k_sub_m_mod_K, W_D_k, W_RF = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        C_m_k = tf.linalg.matmul(T0, Lambda_U_k_sub_m_mod_K, adjoint_a=False, adjoint_b=False)
        R = self.sigma2 * tf.linalg.matmul(C_m_k, C_m_k, adjoint_a=False, adjoint_b=True)
        return R

    # @tf.function
    # def Rq_per_k(self, bundeled_inputs_0):
    #     V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0  # [k, ...]
    #     RQ = tf.zeros(shape=[self.N_s, self.N_s], dtype=tf.float32)
    #     for m in tf.range(self.K):
    #         RQ = tf.add(RQ, tf.add(
    #             tf.cast(self.R_N_Q_m_k([Lambda_U[tf.math.floormod(k - m, self.K), :], W_D[k, :], W_RF]), tf.float32),
    #             tf.cast(self.R_I_Q_m_k([V_D[m, :], W_D[k, :], H, V_RF, W_RF, Lambda_B, Lambda_U, k, m]), tf.float32)))
    #     return RQ

    @tf.function
    def non_zero_element_finder_for_C_m(self, k, truncation_ratio_keep):
        z = 1 - truncation_ratio_keep
        B_orig = int(
            self.K / 2. - z * self.K / 2.)  # original position of zero starting in the fft sequence of phase noise
        ZI = tf.math.floormod(B_orig + tf.range(int(self.K * z)),
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

        return mask_of_ones_after_shift_flip_true

    @tf.function
    def Rq_per_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0  # [k, ...]
        # RIQ = tf.zeros(shape=[self.N_s, self.N_s], dtype=tf.float32)
        # for m in tf.range(self.K):
        #     RIQ = tf.add(RIQ, tf.cast(self.R_I_Q_m_k([V_D[m, :], W_D[k, :], H, V_RF, W_RF, Lambda_B, Lambda_U, k, m]),
        #                               tf.float32))

        W_D_k_repeated = tf.tile([W_D[k, :]], multiples=[self.K, 1, 1])
        H_repeated = tf.tile([H], multiples=[self.K, 1, 1, 1])
        V_RF_repeated = tf.tile([V_RF], multiples=[self.K, 1, 1])
        W_RF_repeated = tf.tile([W_RF], multiples=[self.K, 1, 1])
        Lambda_B_repeated = tf.tile([Lambda_B], multiples=[self.K, 1, 1, 1])
        Lambda_U_repeated = tf.tile([Lambda_U], multiples=[self.K, 1, 1, 1])
        k_repeated = tf.tile([k], multiples=[self.K])
        all_m = tf.reshape(tf.range(self.K), shape=[self.K])
        bundeled_inputs_1 = [V_D, W_D_k_repeated, H_repeated, V_RF_repeated,
                             W_RF_repeated, Lambda_B_repeated, Lambda_U_repeated, k_repeated, all_m]

        RIQ = tf.reduce_sum(tf.map_fn(self.R_I_Q_m_k,
                                bundeled_inputs_1,
                                fn_output_signature=tf.complex64,
                                parallel_iterations=self.K), axis=0)

        mask_of_ones = self.non_zero_element_finder_for_C_m(k, self.truncation_ratio_keep)
        # todo: nozero element finder bypassed
        # mask_of_ones = tf.range(self.K)
        Lambda_U_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, tf.squeeze(k), flip=True),
                                                      mask=mask_of_ones, axis=0)
        W_D_k_repeated = tf.tile([W_D[k, :]], multiples=[round(self.truncation_ratio_keep * self.K), 1, 1])
        W_RF_repeated = tf.tile([W_RF], multiples=[round(self.truncation_ratio_keep * self.K), 1, 1])
        bundeled_inputs_2 = [Lambda_U_cyclshifted_masked, W_D_k_repeated, W_RF_repeated]
        RNQ = tf.reduce_sum(tf.map_fn(self.R_N_Q_m_k,
                                      bundeled_inputs_2,
                                      fn_output_signature=tf.complex64,
                                      parallel_iterations=round(self.K * self.truncation_ratio_keep)),
                            axis=0)

        return tf.add(tf.cast(RIQ, tf.float32), tf.cast(RNQ, tf.float32))

    @tf.function
    def capacity_and_RX_RQ_per_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0  # [k, ...]
        RX = self.Rx_per_k([V_D[k, :], W_D[k, :], H, V_RF, W_RF, Lambda_B, Lambda_U, k])
        RQ = self.Rq_per_k([V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k])

        precision_fixer = 1e-10
        RQ = tf.add(precision_fixer * tf.eye(self.N_s, dtype=tf.float32), RQ)  # numeric precision flaw
        T0 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.float32), tf.linalg.det(RQ)),
                     lambda: tf.multiply(tf.zeros([1], dtype=tf.float32), RQ),
                     lambda: tf.linalg.inv(RQ))
        T1 = tf.linalg.matmul(T0, RX, adjoint_a=False, adjoint_b=False)
        T2 = tf.add(tf.eye(self.N_s, dtype=tf.float32), T1)
        T3 = tf.math.real(tf.linalg.det(T2))
        eta = 0.
        return tf.cond(tf.less(0.0, T3),
                       lambda: tf.divide(tf.math.log(T3), tf.math.log(2.0)),
                       lambda: tf.multiply(eta, T3)), RX, RQ

    @tf.function
    def capacity_forall_k(self, bundeled_inputs_0):
        k_vec = tf.convert_to_tensor(
            np.random.choice(self.K, round(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False),
            dtype=tf.int32)

        if (self.impl == 'map_fn'):
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
            V_D = tf.tile([V_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            W_D = tf.tile([W_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            H = tf.tile([H], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            V_RF = tf.tile([V_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
            W_RF = tf.tile([W_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
            Lambda_B = tf.tile([Lambda_B],
                               multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            Lambda_U = tf.tile([Lambda_U],
                               multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k_vec]
            c, RX, RQ = tf.map_fn(self.capacity_and_RX_RQ_per_k,
                                  bundeled_inputs_1,
                                  fn_output_signature=(tf.float32, tf.float32, tf.float32),
                                  parallel_iterations=self.K)
            return tf.reduce_mean(c, axis=0), RX, RQ
        elif (self.impl == 'vectorized_map'):
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
            V_D = tf.tile([V_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            W_D = tf.tile([W_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            H = tf.tile([H], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            V_RF = tf.tile([V_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
            W_RF = tf.tile([W_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
            Lambda_B = tf.tile([Lambda_B],
                               multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            Lambda_U = tf.tile([Lambda_U],
                               multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
            bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k_vec]
            c, RX, RQ = tf.vectorized_map(self.capacity_and_RX_RQ_per_k,
                                          bundeled_inputs_1)
            return tf.reduce_mean(c, axis=0), RX, RQ
        else:
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
            c_tmp2 = []
            RX_tmp2 = []
            RQ_tmp2 = []
            for k in k_vec:
                T = self.capacity_and_RX_RQ_per_k([V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k])
                c_tmp2.append(T[0])
                RX_tmp2.append(T[1])
                RQ_tmp2.append(T[2])
            c = tf.stack(c_tmp2, axis=0)
            RX = tf.stack(RX_tmp2, axis=0)
            RQ = tf.stack(RQ_tmp2, axis=0)
            return tf.reduce_mean(c, axis=0), RX, RQ

    @tf.function
    def capacity_forall_symbols(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [Nsymb, k, ...]
        # print(H)
        H = tf.tile([H], multiples=[round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1])
        bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U]
        # print('in phase noised loss function =', bundeled_inputs_1)
        c, RX, RQ = tf.map_fn(self.capacity_forall_k, bundeled_inputs_1,
                                  fn_output_signature=(tf.float32, tf.float32, tf.float32),
                                  parallel_iterations=round(self.Nsymb * self.sampling_ratio_time_domain_keep))

        return c, RX, RQ

    @tf.function
    def capacity_forall_samples(self, bundeled_inputs_0):
        if (self.impl == 'map_fn'):
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [batch, Nsymb, k, ...]
            c, RX, RQ = tf.map_fn(self.capacity_forall_symbols, [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U],
                                  fn_output_signature=(tf.float32, tf.float32, tf.float32),
                                  parallel_iterations=self.BATCHSIZE)
            return tf.multiply(-1.0, tf.reduce_mean(tf.reduce_mean(c, axis=0), axis=0)), c, RX, RQ
        elif (self.impl == 'vectorized_map'):
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [batch, Nsymb, k, ...]
            c, RX, RQ = tf.vectorized_map(self.capacity_forall_symbols, [V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U])
            return tf.multiply(-1.0, tf.reduce_mean(tf.reduce_mean(c, axis=0), axis=0)), c, RX, RQ
        else:
            V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [batch, Nsymb, k, ...]
            c_tmp0 = []
            RX_tmp0 = []
            RQ_tmp0 = []
            for ij in tf.range(self.BATCHSIZE):
                T = self.capacity_forall_symbols([V_D[ij, :],
                                                  W_D[ij, :],
                                                  H[ij, :],
                                                  V_RF[ij, :],
                                                  W_RF[ij, :],
                                                  Lambda_B[ij, :],
                                                  Lambda_U[ij, :]])
                c_tmp0.append(T[0])
                RX_tmp0.append(T[1])
                RQ_tmp0.append(T[2])
            c = tf.stack(c_tmp0, axis=0)
            RX = tf.stack(RX_tmp0, axis=0)
            RQ = tf.stack(RQ_tmp0, axis=0)
            return -1.0 * tf.reduce_mean(tf.reduce_mean(c, axis=0), axis=0), c, RX, RQ
