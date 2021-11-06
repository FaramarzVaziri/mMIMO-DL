import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, Reshape, MaxPooling3D, BatchNormalization, Input, Flatten, Dropout, \
    Activation, Add, Conv3DTranspose, Multiply, Lambda, Embedding

# global ofdm_symbol_id
import numpy as np


#
# # https://research.fb.com/wp-content/uploads/2017/01/paper_expl_norm_on_deep_res_networks.pdf
class IdentityBlock_v2_class(tf.keras.Model):  #
    def __init__(self, filters, kernel_size, strides, dilation, trainable):
        super(IdentityBlock_v2_class, self).__init__(name='')
        self.bn0 = BatchNormalization(trainable=trainable)
        self.conv1 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn1 = BatchNormalization(trainable=trainable)

        self.conv2 = Conv3D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            strides=strides,
                            padding='same',
                            dilation_rate=dilation,
                            trainable=trainable)
        self.bn2 = BatchNormalization(trainable=trainable)

        self.act = Activation('relu')

        # residual connection
        self.res_con = Conv3D(filters=filters,
                              kernel_size=(1, 1, 1),
                              strides=(1, 1, 1),
                              activation=None,
                              padding='same',
                              dilation_rate=dilation,
                              trainable=trainable)

        self.add = Add()

    def call(self, input_tensor):
        y = self.conv1(input_tensor)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        z = self.add([y, self.res_con(input_tensor)])
        # z = self.act(z)
        return z


class ResNet_model_class():
    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, SNR, P, N_c, N_scatterers,
                 angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, mat_fname,
                 dataset_size, dropout_rate, convolutional_kernels, convolutional_filters,
                 convolutional_strides, convolutional_dilation):
        self.N_b_a = N_b_a
        self.N_b_rf = N_b_rf
        self.N_u_a = N_u_a
        self.N_u_rf = N_u_rf
        self.N_s = N_s
        self.K = K
        self.PTRS_seperation = PTRS_seperation
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
        self.dropout_rate = dropout_rate
        self.convolutional_kernels = convolutional_kernels
        self.convolutional_filters = convolutional_filters
        self.convolutional_strides = convolutional_strides
        self.convolutional_dilation = convolutional_dilation

    def resnet_function_transceiver(self, trainable_csi, trainable_ns):
        kernels = [min(self.K, self.convolutional_kernels),
                   min(self.N_u_a, self.convolutional_kernels),
                   min(self.N_b_a, self.convolutional_kernels)]
        csi = Input(shape=(self.K, self.N_u_a, self.N_b_a, 2), batch_size=self.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.BATCHSIZE)

        # common path
        ID_block_common_branch_layer_1 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                                kernel_size=kernels,
                                                                strides=self.convolutional_strides,
                                                                dilation=self.convolutional_dilation,
                                                                trainable=trainable_csi)
        ID_block_common_branch_layer_2 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                                kernel_size=kernels,
                                                                strides=self.convolutional_strides,
                                                                dilation=self.convolutional_dilation,
                                                                trainable=trainable_csi)
        ID_block_common_branch_layer_3 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                                kernel_size=kernels,
                                                                strides=self.convolutional_strides,
                                                                dilation=self.convolutional_dilation,
                                                                trainable=trainable_csi)

        ID_block_common_branch_layer_4 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                                kernel_size=kernels,
                                                                strides=self.convolutional_strides,
                                                                dilation=self.convolutional_dilation,
                                                                trainable=trainable_csi)
        # ns path using embedding
        # https: // www.youtube.com / watch?v = VkjSaOZSZVs

        Embedding_layer_cos_D = Embedding(input_dim=self.Nsymb,
                                          output_dim=self.K * self.N_u_a * self.N_b_a * 1,
                                          trainable=trainable_ns)
        Embedding_layer_sin_D = Embedding(input_dim=self.Nsymb,
                                          output_dim=self.K * self.N_u_a * self.N_b_a * 1,
                                          trainable=trainable_ns)
        reshaper_Delta_cos_D = Reshape(target_shape=[self.K, self.N_u_a, self.N_b_a, 1])
        reshaper_Delta_sin_D = Reshape(target_shape=[self.K, self.N_u_a, self.N_b_a, 1])
        add_D = Add()


        Embedding_layer_RF = Embedding(input_dim=self.Nsymb,
                                          output_dim=self.K * self.N_u_a * self.N_b_a * 1,
                                          trainable=trainable_ns)
        reshaper_Delta_RF = Reshape(target_shape=[self.K, self.N_u_a, self.N_b_a, 1])

        add_RF = Add()
        # FC_ns = Dense(units= self.K*self.N_u_a*self.N_b_a*2, activation='relu', use_bias=True,
        #               kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
        #               bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        #

        # V_D path
        ID_block_D_branch_layer_1 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_2 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_3 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_4 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_5 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_6 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_7 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_8 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        ID_block_D_branch_layer_9 = IdentityBlock_v2_class(filters=2,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi)
        MP_D_branch_end = MaxPooling3D(pool_size=(1, int(self.N_u_a / self.N_b_rf), int(self.N_b_a / self.N_s)))
        reshaper_D = Reshape(target_shape=[self.K, self.N_b_rf, self.N_s, 2])

        # V_RF path
        ID_block_RF_branch_layer_1 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_2 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_3 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_4 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_5 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_6 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_7 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_8 = IdentityBlock_v2_class(filters=self.convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        ID_block_RF_branch_layer_9 = IdentityBlock_v2_class(filters=1,
                                                            kernel_size=kernels,
                                                            strides=self.convolutional_strides,
                                                            dilation=self.convolutional_dilation,
                                                            trainable=trainable_csi)
        MP_RF_branch_end = MaxPooling3D(pool_size=(self.K, int(self.N_u_a / self.N_b_a), self.N_b_a))


        # connections
        x = ID_block_common_branch_layer_1(csi)
        x = ID_block_common_branch_layer_2(x)
        x = ID_block_common_branch_layer_3(x)
        x = ID_block_common_branch_layer_4(x)

        # # ns path
        # V_D path
        vd = ID_block_D_branch_layer_1(x)
        vd = ID_block_D_branch_layer_2(vd)
        vd = ID_block_D_branch_layer_3(vd)
        vd = ID_block_D_branch_layer_4(vd)
        vd = ID_block_D_branch_layer_5(vd)
        vd = ID_block_D_branch_layer_6(vd)
        vd = ID_block_D_branch_layer_7(vd)
        vd = ID_block_D_branch_layer_8(vd)
        vd = ID_block_D_branch_layer_9(vd)
        vd = add_D([vd, tf.concat([reshaper_Delta_cos_D(Embedding_layer_cos_D(tf.cast(ns, tf.float32))),
                                   reshaper_Delta_sin_D(Embedding_layer_sin_D(tf.cast(ns, tf.float32)))],
                                axis=4)])
        vd_end_out = MP_D_branch_end(vd)
        vd_end = reshaper_D(vd_end_out)

        # V_RF path
        vrf = ID_block_RF_branch_layer_1(x)
        vrf = ID_block_RF_branch_layer_2(vrf)
        vrf = ID_block_RF_branch_layer_3(vrf)
        vrf = ID_block_RF_branch_layer_4(vrf)
        vrf = ID_block_RF_branch_layer_5(vrf)
        vrf = ID_block_RF_branch_layer_6(vrf)
        vrf = ID_block_RF_branch_layer_7(vrf)
        vrf = ID_block_RF_branch_layer_8(vrf)
        vrf = ID_block_RF_branch_layer_9(vrf)
        vrf = add_RF([vrf, reshaper_Delta_RF(Embedding_layer_RF(tf.cast(ns, tf.float32)))])
        vrf_end = MP_RF_branch_end(vrf)

        func_model = Model(inputs=[csi, ns], outputs=[vd_end, vrf_end])
        return func_model

    @tf.function
    def custom_actication_transmitter(self, inputs):
        V_D, vrf = inputs

        V_D_cplx = tf.complex(tf.cast(V_D[:, :, :, :, 0], tf.float32), tf.cast(V_D[:, :, :, :, 1], tf.float32))
        vrf_cplx = tf.complex(tf.cast(tf.cos(vrf), tf.float32), tf.cast(tf.sin(vrf), tf.float32))

        # partially-connected analog beamformer matrix implementation ----------------

        bundeled_inputs_0 = [V_D_cplx, vrf_cplx]
        V_D_new_cplx, V_RF_cplx = tf.map_fn(self.custorm_activation_per_sample_transmitter, bundeled_inputs_0,
                                            fn_output_signature=(tf.complex64, tf.complex64),
                                            parallel_iterations=self.BATCHSIZE)
        return V_D_new_cplx, V_RF_cplx

    @tf.function
    def normalize_power_per_subcarrier_transmitter(self, bundeled_inputs_0):
        V_D_k, V_RF = bundeled_inputs_0
        T0 = tf.linalg.matmul(V_RF, V_D_k, adjoint_a=False, adjoint_b=False)
        T1 = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
        # denum = tf.linalg.trace(T1) #
        denum = tf.add(tf.linalg.trace(T1), tf.complex(1e-16, 1e-16))  ###### numeric precision flaw
        V_D_k_normalized = tf.divide(tf.multiply(V_D_k, tf.cast(tf.sqrt(self.P), dtype=tf.complex64)), tf.sqrt(denum))
        return V_D_k_normalized

    @tf.function
    def custorm_activation_per_sample_transmitter(self, bundeled_inputs_0):
        V_D_cplx, vrf_cplx = bundeled_inputs_0
        # for BS
        vrf_zero_padded = tf.concat([tf.reshape(vrf_cplx, shape=[self.N_b_a, 1]),
                                     tf.zeros(shape=[self.N_b_a, self.N_b_rf - 1], dtype=tf.complex64)], axis=1)
        r_bs = int(self.N_b_a / self.N_b_rf)
        T2_BS = []
        for i in range(self.N_b_rf):
            T0_BS = vrf_zero_padded[r_bs * i: r_bs * (i + 1), :]
            T1_BS = tf.roll(T0_BS, shift=i, axis=1)
            T2_BS.append(T1_BS)
        V_RF_per_sample = tf.concat(T2_BS, axis=0)

        # repeating inputs for vectorization
        V_RF_per_sample_repeated_K_times = tf.tile([V_RF_per_sample], multiples=[self.K, 1, 1])
        bundeled_inputs_1 = [V_D_cplx, V_RF_per_sample_repeated_K_times]
        V_D_cplx_normalized_per_sample = tf.map_fn(self.normalize_power_per_subcarrier_transmitter, bundeled_inputs_1,
                                                   fn_output_signature=tf.complex64, parallel_iterations=self.K)

        return V_D_cplx_normalized_per_sample, V_RF_per_sample

    @tf.function
    def custom_actication_receiver(self, inputs0):
        W_D, wrf = inputs0
        W_D_cplx = tf.complex(tf.cast(W_D[:, :, :, :, 0], tf.float32), tf.cast(W_D[:, :, :, :, 1], tf.float32))
        wrf_cplx = tf.complex(tf.cast(tf.cos(wrf), tf.float32), tf.cast(tf.sin(wrf), tf.float32))

        # partially-connected analog beamformer matrix implementation ----------------

        W_RF_cplx = tf.map_fn(self.custorm_activation_per_sample_receiver, wrf_cplx,
                              fn_output_signature=(tf.complex64),
                              parallel_iterations=self.BATCHSIZE)

        return W_D_cplx, W_RF_cplx

    @tf.function
    def custorm_activation_per_sample_receiver(self, wrf_cplx):
        # for UE
        wrf_zero_padded = tf.concat([tf.reshape(wrf_cplx, shape=[self.N_u_a, 1]),
                                     tf.zeros(shape=[self.N_u_a, self.N_u_rf - 1], dtype=tf.complex64)], axis=1)
        r_ue = int(self.N_u_a / self.N_u_rf)
        T2_UE = []
        for i in range(self.N_u_rf):
            T0_UE = wrf_zero_padded[r_ue * i: r_ue * (i + 1), :]
            T1_UE = tf.roll(T0_UE, shift=i, axis=1)
            T2_UE.append(T1_UE)
        W_RF_per_sample = tf.concat(T2_UE, axis=0)

        return W_RF_per_sample
