
import tensorflow as tf
import numpy as np

class loss_phase_noise_free_class:


    def __init__(self,N_b_a,N_b_rf,N_u_a,N_u_rf,N_s,K,SNR,P,N_c,N_scatterers,angular_spread_rad,wavelength,d,
                 BATCHSIZE,Nsymb, sampling_ratio_time_domain_keep,impl,sampling_ratio_subcarrier_domain_keep):
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
        self.Nsymb = Nsymb
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.impl = impl
        self.sampling_ratio_subcarrier_domain_keep = sampling_ratio_subcarrier_domain_keep

    @tf.function
    def C_per_sample_per_k(self,bundeled_inputs):
        V_D_cplx, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, k = bundeled_inputs  # no vectorization
        T0 = tf.linalg.matmul(W_D_cplx[k, :], W_RF_cplx, adjoint_a=True, adjoint_b=True)
        T1 = tf.linalg.matmul(T0, H_complex[k, :], adjoint_a=False, adjoint_b=False)
        T2 = tf.linalg.matmul(T1, V_RF_cplx, adjoint_a=False, adjoint_b=False)
        T3 = tf.linalg.matmul(T2, V_D_cplx[k, :], adjoint_a=False, adjoint_b=False)
        R_X = tf.linalg.matmul(T3, T3, adjoint_a=False, adjoint_b=True)
        R_Q = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
        T4 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.complex64), tf.linalg.det(R_Q)),
                     lambda: tf.multiply(tf.zeros([1], dtype=tf.complex64), R_Q), lambda: tf.linalg.inv(R_Q))
        # T4 = tf.linalg.inv(R_Q)
        T5 = tf.linalg.matmul(T4, R_X, adjoint_a=False, adjoint_b=False)
        T6 = tf.add(tf.eye(self.N_s, dtype=tf.complex64), tf.divide(T5, tf.cast(self.sigma2, dtype=tf.complex64)))
        T7 = tf.math.real(tf.linalg.det(T6))
        eta = 0.
        # T8 = tf.cond(tf.less(0.0 , T7), lambda: tf.divide(tf.math.log( T7 ) , tf.math.log(2.0)), lambda: tf.multiply(eta , T7))
        T8 = tf.divide(tf.math.log(T7), tf.math.log(2.0))
        return T8


    @tf.function
    def capacity_forall_k(self, bundeled_inputs_0):
        k_vec = tf.convert_to_tensor(
            np.random.choice(self.K, round(self.sampling_ratio_subcarrier_domain_keep * self.K), replace=False),
            dtype=tf.int32)
        V_D, W_D, H, V_RF, W_RF = bundeled_inputs_0  # [k, ...]
        V_D = tf.tile([V_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        W_D = tf.tile([W_D], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        H = tf.tile([H], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1, 1])
        V_RF = tf.tile([V_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
        W_RF = tf.tile([W_RF], multiples=[round(self.sampling_ratio_subcarrier_domain_keep * self.K), 1, 1])
        bundeled_inputs_vectorized_on_k = [V_D, W_D, H, V_RF, W_RF, k_vec]
        T0 = tf.map_fn(self.C_per_sample_per_k, bundeled_inputs_vectorized_on_k,
                           fn_output_signature=tf.float32) # , parallel_iterations=self.K
        return tf.reduce_mean(T0)

    @tf.function
    def capacity_forall_symbols(self, bundeled_inputs_0):
        V_D, W_D, H, V_RF, W_RF = bundeled_inputs_0  # [Nsymb, k, ...]
        H = tf.tile([H], multiples=[round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1])
        bundeled_inputs_1 = [V_D, W_D, H, V_RF, W_RF]
        # print('bundeled_inputs_1 =', bundeled_inputs_1)
        c = tf.reduce_mean(tf.map_fn(self.capacity_forall_k, bundeled_inputs_1,
                              fn_output_signature=tf.float32,
                              parallel_iterations = round(self.Nsymb * self.sampling_ratio_time_domain_keep)), axis=0)
        return c

    @tf.function
    def capacity_forall_samples(self, bundeled_inputs):
        if (self.impl == 'map_fn'):
            T0 = tf.map_fn(self.capacity_forall_symbols, bundeled_inputs, fn_output_signature=tf.float32, parallel_iterations=self.BATCHSIZE) #
            return tf.multiply(-1.0, tf.reduce_mean(T0))
        # elif (self.impl == 'vectorized_map'):
        #     V_D_cplx, W_D_cplx, H, V_RF_cplx, W_RF_cplx = bundeled_inputs
        #     H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        #     bundeled_inputs_modified = [V_D_cplx, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx]
        #     T0 = tf.vectorized_map(self.C_per_sample, bundeled_inputs_modified)
        #     return tf.multiply(-1.0, tf.reduce_mean(T0))
        # else:
        #     V_D_cplx, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx = bundeled_inputs
        #     # H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        #     T0 = tf.zeros(shape = [1], dtype=tf.float32)
        #     for ij in tf.range(self.BATCHSIZE):
        #         T0 = tf.add(T0, self.C_per_sample( [V_D_cplx[ij,:],
        #                                             W_D_cplx[ij,:],
        #                                             H_complex[ij,:],
        #                                             V_RF_cplx[ij,:],
        #                                             W_RF_cplx[ij,:]]))
        #     return -1.0 * T0 / self.BATCHSIZE

