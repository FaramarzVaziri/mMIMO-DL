import numpy as np
import tensorflow as tf


class sequential_loss_phase_noised_class:

    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, truncation_ratio_keep, Nsymb,
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
        mask_of_ones_after_shift_flip_false = tf.roll(mask_of_ones_before_shift, shift=tf.squeeze(k), axis=0)
        mask_of_ones_after_shift_total = tf.multiply(mask_of_ones_after_shift_flip_true,
                                                     mask_of_ones_after_shift_flip_false)
        return mask_of_ones_after_shift_total

    @tf.function
    def Rx_per_k(self, bundeled_inputs_0):
        V_D_k, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0 # [k, ...]

        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_U_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, k, flip=True ), mask=mask_of_ones, axis=0)
        Lambda_B_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, k, flip=False), mask=mask_of_ones, axis=0)

        H_tilde_k = tf.zeros(shape=[self.N_u_a, self.N_b_a], dtype= tf.complex64)
        for q in range(int(self.K * self.truncation_ratio_keep)):
            H_tilde_k = tf.add(H_tilde_k,
                               tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked[q],
                                                                 H_masked[q]), Lambda_B_cyclshifted_masked[q]))

        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, H_tilde_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        A_ns_k = tf.linalg.matmul(T2, V_D_k)
        RX_k = tf.linalg.matmul(A_ns_k, A_ns_k, adjoint_a=False, adjoint_b=True)
        return RX_k

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
        if (m == k):
            R = tf.zeros(shape= [self.N_s, self.N_s], dtype=tf.complex64)
        else:
            mask_of_ones = self.non_zero_element_finder_for_H_hat(k, m, self.truncation_ratio_keep)
            H_masked = tf.boolean_mask(H, mask= mask_of_ones, axis=0)
            Lambda_U_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, tf.squeeze(k), flip=True), mask= mask_of_ones, axis=0)
            Lambda_B_cyclshifted_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, tf.squeeze(m), flip=False), mask= mask_of_ones, axis=0)

            H_hat_m_k = tf.zeros(shape= [self.N_u_a, self.N_b_a], dtype=tf.complex64)
            for q in range(self.K * self.truncation_ratio_keep):
                H_hat_m_k = tf.add(H_hat_m_k, tf.linalg.matmul(tf.linalg.matmul(Lambda_U_cyclshifted_masked[q,:], H_masked[q,:], adjoint_a=False, adjoint_b=False),
                    Lambda_B_cyclshifted_masked[q,:], adjoint_a=False, adjoint_b=False))

            T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
            T1 = tf.linalg.matmul(T0, H_hat_m_k)
            T2 = tf.linalg.matmul(T1, V_RF)
            B_m_k = tf.linalg.matmul(T2, V_D_m)
            R = tf.linalg.matmul(B_m_k, B_m_k, adjoint_a=False, adjoint_b=True)
        return R

    @tf.function
    def R_N_Q_m_k(self, bundeled_inputs_0):
        Lambda_U_k_sub_m_mod_K, W_D_k, W_RF = bundeled_inputs_0
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        C_m_k = tf.linalg.matmul(T0, Lambda_U_k_sub_m_mod_K, adjoint_a=False, adjoint_b=False)
        R = self.sigma2 * tf.linalg.matmul(C_m_k, C_m_k, adjoint_a=False, adjoint_b=True)
        return R

    @tf.function
    def Rq_per_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0 # [k, ...]
        RQ = tf.zeros(shape= [self.N_s, self.N_s], dtype=tf.complex64)
        for m in range(self.K):
            RQ = tf.add(RQ, tf.add( tf.cast(self.R_N_Q_m_k([Lambda_U[tf.math.floormod(k - m, self.K), :], W_D[k,:], W_RF]), tf.complex64),
                      tf.cast(self.R_I_Q_m_k([V_D[m,:], W_D[k,:], H, V_RF, W_RF, Lambda_B, Lambda_U, k, m]), tf.complex64)))
        return RQ

    @tf.function
    def capacity_and_RX_RQ_per_k(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k = bundeled_inputs_0 # [k, ...]
        RX = self.Rx_per_k([V_D[k,:], W_D[k,:], H, V_RF, W_RF, Lambda_B, Lambda_U, k])
        RQ = self.Rq_per_k([V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k])
        precision_fixer = 1e-7
        RQ = tf.add(precision_fixer * tf.eye(self.N_s, dtype=tf.complex64), RQ)  # numeric precision flaw
        T0 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.complex64), tf.linalg.det(RQ)),
                     lambda: tf.multiply(tf.zeros([1], dtype=tf.complex64), RQ),
                     lambda: tf.linalg.inv(RQ))
        T1 = tf.linalg.matmul(T0, RX, adjoint_a=False, adjoint_b=False)
        T2 = tf.add(tf.eye(self.N_s, dtype=tf.complex64), T1)
        T3 = tf.math.real(tf.linalg.det(T2))
        eta = 0.
        return tf.cond(tf.less(0.0, T3),
                       lambda: tf.divide(tf.math.log(T3), tf.math.log(2.0)),
                       lambda: tf.multiply(eta, T3)), RX, RQ

    @tf.function
    def capacity_forall_k(self, bundeled_inputs_0):
        # # impl with for ------------------------
        # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
        # # sampled_K = tf.convert_to_tensor(np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False),dtype=tf.int64)
        # c = 0.
        # RX_tmp = []
        # RQ_tmp = []
        # # for k in range(self.K):
        # for k in np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False).astype(int):
        #     T = self.capacity_and_RX_RQ_per_k([V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k])
        #     c = c + T[0]/self.K
        #     RX_tmp.append(T[1])
        #     RQ_tmp.append(T[2])
        #
        # RX = tf.stack(RX_tmp, axis=0)
        # RQ = tf.stack(RQ_tmp, axis=0)
        # return c, RX, RQ

    #  impl 2 with for and concat ------------------------
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [k, ...]
        c = 0.
        for k in range(self.K): # np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False).astype(int):
            T = self.capacity_and_RX_RQ_per_k([V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U, k])
            c = c + T[0] / self.K
            if (k==0):
                RX = tf.expand_dims(T[1], axis=0)
                RQ = tf.expand_dims(T[2], axis=0)
            else:
                RX = tf.concat([RX, tf.expand_dims(T[1], axis=0)], axis=0)
                RQ = tf.concat([RQ, tf.expand_dims(T[2], axis=0)], axis=0)

        return c, RX, RQ

    @tf.function
    def capacity_forall_symbols(self, bundeled_inputs_0):
        # #  impl with for
        # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0 # [Nsymb, k, ...]
        # selected_symbols = np.random.choice(self.Nsymb, int(self.sampling_ratio_time_domain_keep * self.Nsymb), replace=False)
        # c_tmp = []
        # RX_tmp = []
        # RQ_tmp = []
        # for ns in range(0, self.Nsymb):
        #     T = self.capacity_forall_k([V_D, W_D, H, V_RF, W_RF, Lambda_B[ns,:], Lambda_U[ns,:]])
        #     c_tmp.append(T[0])
        #     RX_tmp.append(T[1])
        #     RQ_tmp.append(T[2])
        #
        # c = tf.stack(c_tmp, axis=0)
        # RX = tf.stack(RX_tmp, axis=0)
        # RQ = tf.stack(RQ_tmp, axis=0)
        # return c, RX, RQ

        #  impl 2 with for and concat ------------------------
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0 # [Nsymb, k, ...]
        selected_symbols = np.random.choice(self.Nsymb, int(self.sampling_ratio_time_domain_keep * self.Nsymb), replace=False)
        for ns in range(0, self.Nsymb, 10):
            T = self.capacity_forall_k([V_D, W_D, H, V_RF, W_RF, Lambda_B[ns,:], Lambda_U[ns,:]])
            if (ns == 0):
                c = tf.expand_dims(T[0], axis=0)
                RX = tf.expand_dims(T[1], axis=0)
                RQ = tf.expand_dims(T[2], axis=0)
            else:
                c = tf.concat([c, tf.expand_dims(T[0], axis=0)], axis=0)
                RX = tf.concat([RX, tf.expand_dims(T[1], axis=0)], axis=0)
                RQ = tf.concat([RQ, tf.expand_dims(T[2], axis=0)], axis=0)
        return c, RX, RQ

    @tf.function
    def capacity_forall_samples(self, bundeled_inputs_0):
        # # impl with for ------------------------------------------------------------------------------------------------
        # V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0 # [batch, Nsymb, k, ...]
        # c_tmp = []
        # RX_tmp = []
        # RQ_tmp = []
        # for ij in range(self.BATCHSIZE):
        #     T = self.capacity_forall_symbols([V_D[ij,:], W_D[ij,:], H[ij,:], V_RF[ij,:], W_RF[ij,:], Lambda_B[ij,:], Lambda_U[ij,:]])
        #     c_tmp.append(T[0])
        #     RX_tmp.append(T[1])
        #     RQ_tmp.append(T[2])
        #
        # c = tf.stack(c_tmp, axis=0)
        # RX = tf.stack(RX_tmp, axis=0)
        # RQ = tf.stack(RQ_tmp, axis=0)
        #
        # return -1.0*tf.reduce_mean(tf.reduce_mean(c, axis=0), axis=0), c, RX, RQ


        # impl 2 with for and concat -----------------------------------------------------------------------------------
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0  # [batch, Nsymb, k, ...]
        for ij in range(self.BATCHSIZE):
            T = self.capacity_forall_symbols([V_D[ij, :],
                                              W_D[ij, :],
                                              H[ij, :],
                                              V_RF[ij, :],
                                              W_RF[ij, :],
                                              Lambda_B[ij, :],
                                              Lambda_U[ij, :]])
            if (ij == 0):
                c = tf.expand_dims(T[0], axis=0)
                RX = tf.expand_dims(T[1], axis=0)
                RQ= tf.expand_dims(T[2], axis=0)
            else:
                c = tf.concat([c , tf.expand_dims(T[0], axis=0)], axis=0)
                RX = tf.concat([RX, tf.expand_dims(T[1], axis=0)], axis=0)
                RQ = tf.concat([RQ, tf.expand_dims(T[2], axis=0)], axis=0)

        return -1.0 * tf.reduce_mean(tf.reduce_mean(c, axis=0), axis=0), c, RX, RQ