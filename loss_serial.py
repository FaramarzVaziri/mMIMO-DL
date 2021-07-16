import tensorflow as tf

class serial_loss_class:
    def __init__(self,N_b_a,N_b_rf,N_u_a,N_u_rf,N_s,K,SNR,P,N_c,N_scatterers,angular_spread_rad,wavelength,d,BATCHSIZE,phase_shift_stddiv):
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


    def loss_func_custom_serial(self,inputs):
        V_D_cplx, W_D_cplx, H, V_RF_cplx, W_RF_cplx = inputs
        H_complex = tf.complex(H[:, :, :, :, 0], H[:, :, :, :, 1])
        C_tot = tf.zeros(shape=[1], dtype=tf.float32)
        for ij in range(self.BATCHSIZE):
            C = tf.zeros(shape=[1], dtype=tf.float32)
            for k in range(self.K):
                T0 = tf.linalg.matmul(W_D_cplx[ij, k, :, :], W_RF_cplx[ij, :, :], adjoint_a=True, adjoint_b=True)
                T1 = tf.linalg.matmul(T0, H_complex[ij, k, :, :], adjoint_a=False, adjoint_b=False)
                T2 = tf.linalg.matmul(T1, V_RF_cplx[ij, :, :], adjoint_a=False, adjoint_b=False)
                T3 = tf.linalg.matmul(T2, V_D_cplx[ij, k, :, :], adjoint_a=False, adjoint_b=False)
                R_X = tf.linalg.matmul(T3, T3, adjoint_a=False, adjoint_b=True)
                R_Q = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)

                T4 = tf.cond(tf.equal(tf.zeros([1], dtype=tf.complex64), tf.linalg.det(R_Q)),
                             lambda: tf.multiply(tf.zeros([1], dtype=tf.complex64), R_Q), lambda: tf.linalg.inv(R_Q))

                T5 = tf.linalg.matmul(T4, R_X, adjoint_a=False, adjoint_b=False)
                T6 = tf.add(tf.eye(self.N_s, dtype=tf.complex64), tf.divide(T5, tf.cast(self.sigma2, dtype=tf.complex64)))

                T7 = tf.math.real(tf.linalg.det(T6))

                eta = 0.
                # T8 = tf.cond(tf.less(0.0 , T7), lambda: tf.divide(tf.math.log( T7 ) , tf.math.log(2.0)*K), lambda: tf.multiply(eta , T7))
                T8 = tf.divide(tf.math.log(T7), tf.math.log(2.0) * self.K)
                C = tf.add(C, T8)
            C_tot = tf.add(C_tot, C)

        return tf.multiply(-1.0 / self.BATCHSIZE, C_tot)