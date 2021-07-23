import numpy as np
import tensorflow as tf


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

    @tf.function
    def H_tilde_k_calculation(self, H_k, Lambda_B_k, Lambda_U_k):
        T0 = tf.linalg.matmul(Lambda_U_k, H_k)
        T1 = tf.linalg.matmul(T0, Lambda_B_k)
        return T1

    @tf.function
    def Rx_calculation_per_k(self, V_D_k, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k):
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        mask_of_ones = self.non_zero_element_finder_for_H_tilde(k, self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_B_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, k, flip=False),
                                          mask=mask_of_ones, axis=0)
        Lambda_U_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, k, flip=True),
                                          mask=mask_of_ones, axis=0)
        H_tilde_k = 0
        for q in range(int(self.K * self.truncation_ratio_keep)):
            H_tilde_k = H_tilde_k + self.H_tilde_k_calculation(H_masked[q,:], Lambda_B_masked[q,:], Lambda_U_masked[q,:])

        T1 = tf.linalg.matmul(T0, H_tilde_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        A_ns_k = tf.linalg.matmul(T2, V_D_k)
        R_X_k = tf.linalg.matmul(A_ns_k, A_ns_k, adjoint_a=False, adjoint_b=True)
        return R_X_k


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
    def H_hat_m_k_calculation(self, H_k, Lambda_B_k, Lambda_U_k):
        return tf.linalg.matmul(tf.linalg.matmul(Lambda_U_k, H_k), Lambda_B_k)

    @tf.function
    def R_I_Q_m_k(self, H, Lambda_B, Lambda_U, V_D_m, V_RF, W_D_k, W_RF, k, m):
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        mask_of_ones = self.non_zero_element_finder_for_H_hat(tf.squeeze(k), tf.squeeze(m), self.truncation_ratio_keep)
        H_masked = tf.boolean_mask(H, mask=mask_of_ones, axis=0)
        Lambda_B_masked = tf.boolean_mask(self.cyclical_shift(Lambda_B, tf.squeeze(m), flip=False),
                                                   mask=mask_of_ones, axis=0)
        Lambda_U_masked = tf.boolean_mask(self.cyclical_shift(Lambda_U, tf.squeeze(k), flip=True),
                                                   mask=mask_of_ones, axis=0)
        H_hat_m_k = 0
        for q in range(int(self.K * self.truncation_ratio_keep)):
            H_hat_m_k = H_hat_m_k + self.H_hat_m_k_calculation(H_masked[q, :], Lambda_B_masked[q, :], Lambda_U_masked[q, :])

        T1 = tf.linalg.matmul(T0, H_hat_m_k)
        T2 = tf.linalg.matmul(T1, V_RF)
        B_m_k = tf.linalg.matmul(T2, V_D_m)
        R = tf.linalg.matmul(B_m_k, B_m_k, adjoint_a=False, adjoint_b=True)
        return R

    @tf.function
    def R_N_Q_m_k(self, Lambda_U, W_D_k, W_RF, k, m):
        T0 = tf.linalg.matmul(W_D_k, W_RF, adjoint_a=True, adjoint_b=True)
        # size_Lambda_U = Lambda_U.shape
        Lambda_U_for_k_subtract_m = tf.reshape(tf.slice(
            Lambda_U, begin=[tf.math.floormod(tf.squeeze(tf.cast(k, dtype = tf.int32)) - tf.squeeze(tf.cast(m, dtype = tf.int32)), self.K), 0, 0], size=[1, self.N_u_a, self.N_u_a]),
            shape=[self.N_u_a, self.N_u_a])
        C_m_k = tf.linalg.matmul(T0, Lambda_U_for_k_subtract_m)
        R = self.sigma2 * tf.linalg.matmul(C_m_k, C_m_k, adjoint_a=False, adjoint_b=True)
        return R


    @tf.function
    def Rq_calculation_per_k(self, V_D, W_D_k, H, V_RF, W_RF, Lambda_B, Lambda_U, k):
        R_Q  = 0
        for m in range(self.K):
            T0 = tf.cond(tf.equal(tf.cast(m, dtype = tf.int32), tf.cast(k, dtype = tf.int32)),
                         lambda: self.R_N_Q_m_k(Lambda_U, W_D_k, W_RF, k, m),
                         lambda: self.R_I_Q_m_k(H, Lambda_B, Lambda_U, V_D[m,:], V_RF, W_D_k, W_RF, k, m) +
                                 self.R_N_Q_m_k(Lambda_U, W_D_k, W_RF, k, m))
            R_Q = R_Q + T0
        return R_Q


    # Capacity calculation
    @tf.function
    def capacity_calculation_per_k(self, R_X, R_Q):
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
    def capacity_calculation_forall_k(self, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U):
        # # impl with for ---------------------------------------------------------------------------------------------
        sampled_K = np.random.choice(self.K, int(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False, )

        C = 0.0
        RX_forall_k_tmp = []
        RQ_forall_k_tmp = []
        for k in sampled_K.astype(int):
            rx = self.Rx_calculation_per_k(V_D[k, :], W_D[k, :], H, V_RF, W_RF, Lambda_B, Lambda_U, k)
            rq = self.Rq_calculation_per_k(V_D, W_D[k,:], H, V_RF, W_RF, Lambda_B, Lambda_U, k)
            RX_forall_k_tmp.append(rx)
            RQ_forall_k_tmp.append(rq)
            C = C + self.capacity_calculation_per_k(rx, rq)
        RX_forall_k = tf.stack(RX_forall_k_tmp, axis=0)
        RQ_forall_k = tf.stack(RQ_forall_k_tmp, axis=0)
        return C/int(self.sampling_ratio_subcarrier_domain_keep * self.K), RX_forall_k, RQ_forall_k


    @tf.function
    def capacity_calculation_for_frame(self, V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U):

        # # impl with for ----------------------------------------------------------------------------------------------
        selected_symbols = np.random.choice(self.Nsymb, int(self.sampling_ratio_time_domain_keep * self.Nsymb), replace=False)
        ergodic_capacity_forall_OFDMs_tmp = []
        RX_forall_k_forall_OFDMs_tmp = []
        RQ_forall_k_forall_OFDMs_tmp = []
        for ns in selected_symbols.astype(int):
            T = self.capacity_calculation_forall_k(V_D, W_D, H, V_RF, W_RF, Lambda_B[ns,:], Lambda_U[ns,:])
            ergodic_capacity_forall_OFDMs_tmp.append(T[0])
            RX_forall_k_forall_OFDMs_tmp.append(T[1])
            RQ_forall_k_forall_OFDMs_tmp.append(T[2])

        ergodic_capacity_forall_OFDMs = tf.stack(ergodic_capacity_forall_OFDMs_tmp, axis=0)
        RX_forall_k_forall_OFDMs = tf.stack(RX_forall_k_forall_OFDMs_tmp, axis=0)
        RQ_forall_k_forall_OFDMs = tf.stack(RQ_forall_k_forall_OFDMs_tmp, axis=0)
        capacity_sequence_in_frame = tf.reshape(ergodic_capacity_forall_OFDMs,
                                                shape=[1, int(self.sampling_ratio_time_domain_keep * self.Nsymb)])
        return capacity_sequence_in_frame, RX_forall_k_forall_OFDMs, RQ_forall_k_forall_OFDMs


    @tf.function
    def capacity_calculation_for_frame_for_batch(self, bundeled_inputs_0):

        # impl with for ------------------------------------------------------------------------------------------------
        V_D, W_D, H, V_RF, W_RF, Lambda_B, Lambda_U = bundeled_inputs_0
        capacity_sequence_in_frame_forall_samples_tmp = []
        RX_forall_k_forall_OFDMs_forall_samples_tmp = []
        RQ_forall_k_forall_OFDMs_forall_samples_tmp = []
        for ij in range(self.BATCHSIZE):
            T = self.capacity_calculation_for_frame(V_D[ij,:], W_D[ij,:], H[ij,:], V_RF[ij,:], W_RF[ij,:], Lambda_B[ij,:], Lambda_U[ij,:])
            capacity_sequence_in_frame_forall_samples_tmp.append(T[0])
            RX_forall_k_forall_OFDMs_forall_samples_tmp.append(T[1])
            RQ_forall_k_forall_OFDMs_forall_samples_tmp.append(T[2])

        capacity_sequence_in_frame_forall_samples = tf.stack(capacity_sequence_in_frame_forall_samples_tmp, axis=0)
        RX_forall_k_forall_OFDMs_forall_samples = tf.stack(RX_forall_k_forall_OFDMs_forall_samples_tmp, axis=0)
        RQ_forall_k_forall_OFDMs_forall_samples = tf.stack(RQ_forall_k_forall_OFDMs_forall_samples_tmp, axis=0)

        return -1.0*tf.reduce_mean(capacity_sequence_in_frame_forall_samples, axis=0), \
               capacity_sequence_in_frame_forall_samples, RX_forall_k_forall_OFDMs_forall_samples, RQ_forall_k_forall_OFDMs_forall_samples
