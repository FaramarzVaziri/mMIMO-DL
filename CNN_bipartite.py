import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, Reshape, MaxPooling3D, BatchNormalization, Input, Flatten, Dropout, \
    Activation, Add, Conv3DTranspose, Multiply, Lambda, AveragePooling3D

global ofdm_symbol_id
import numpy as np


#
# # https://research.fb.com/wp-content/uploads/2017/01/paper_expl_norm_on_deep_res_networks.pdf
class ResNet_class(tf.keras.Model):  #
    def __init__(self, filters, kernel_size, strides, dilation, trainable):
        super(ResNet_class, self).__init__()
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


class Generic_2_ns_4_large_MIMOFDM_CNN_class(tf.keras.Model):  #
    def __init__(self, convolutional_filters, kernels, kernels_start_and_end, convolutional_strides,
                 convolutional_dilation, trainablity, subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1,
                 subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2,
                 n_common_layers, n_D_and_RF_layers_generic, layer_name):
        super(Generic_2_ns_4_large_MIMOFDM_CNN_class, self).__init__(name=layer_name)
        self.n_common_layers = n_common_layers
        self.n_D_and_RF_layers_generic = n_D_and_RF_layers_generic

        # common path
        self.AP_1 = AveragePooling3D(pool_size=(subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1))
        self.AP_2 = AveragePooling3D(pool_size=(subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2))

        self.ResNet_block_common_branch = []
        self.ResNet_block_common_branch.append(
            ResNet_class(filters=round(convolutional_filters / (subcarrier_strides_l1 * subcarrier_strides_l1)),
                         kernel_size=kernels_start_and_end,
                         strides=convolutional_strides,
                         dilation=convolutional_dilation,
                         trainable=trainablity))
        self.ResNet_block_common_branch.append(
            ResNet_class(filters=round(convolutional_filters / subcarrier_strides_l2),
                         kernel_size=kernels,
                         strides=convolutional_strides,
                         dilation=convolutional_dilation,
                         trainable=trainablity))

        for i in range(2, n_common_layers, 1):
            self.ResNet_block_common_branch.append(ResNet_class(filters=convolutional_filters,
                                                                kernel_size=kernels,
                                                                strides=convolutional_strides,
                                                                dilation=convolutional_dilation,
                                                                trainable=trainablity))

        # V_D path
        self.ResNet_block_D_branch = []
        for i in range(n_D_and_RF_layers_generic):
            self.ResNet_block_D_branch.append(ResNet_class(filters=convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=convolutional_strides,
                                                           dilation=convolutional_dilation,
                                                           trainable=trainablity))

        # V_RF path
        self.ResNet_block_RF_branch = []
        for i in range(n_D_and_RF_layers_generic):
            self.ResNet_block_RF_branch.append(ResNet_class(filters=convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=convolutional_strides,
                                                            dilation=convolutional_dilation,
                                                            trainable=trainablity))

    def call(self, csi):
        # connections
        x = self.ResNet_block_common_branch[0](csi)
        x = self.AP_1(x)
        x = self.ResNet_block_common_branch[1](x)
        x = self.AP_2(x)
        for i in range(2, self.n_common_layers, 1):
            x = self.ResNet_block_common_branch[i](x)

        # V_D path
        vd = self.ResNet_block_D_branch[0](x)
        for i in range(1, self.n_D_and_RF_layers_generic, 1):
            vd = self.ResNet_block_D_branch[i](vd)

        # V_RF path
        vrf = self.ResNet_block_RF_branch[0](x)
        for i in range(1, self.n_D_and_RF_layers_generic, 1):
            vrf = self.ResNet_block_RF_branch[i](vrf)

        return vd, vrf


class Specialized_2_ns_4_large_MIMOFDM_CNN_class(tf.keras.Model):  #
    def __init__(self, convolutional_filters, kernels, kernels_start_and_end, convolutional_strides,
                 convolutional_dilation,
                 N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, trainablity,
                 subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1,
                 subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2,
                 n_D_and_RF_layers_specialized, layer_name):
        super(Specialized_2_ns_4_large_MIMOFDM_CNN_class, self).__init__(name=layer_name)
        self.n_D_and_RF_layers_specialized = n_D_and_RF_layers_specialized

        # V_D path
        self.ResNet_block_D_branch = []
        for i in range(self.n_D_and_RF_layers_specialized):
            self.ResNet_block_D_branch.append(ResNet_class(filters=convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=convolutional_strides,
                                                           dilation=convolutional_dilation,
                                                           trainable=trainablity))

        self.Tconv_D_1 = Conv3DTranspose(filters=convolutional_filters,
                                         kernel_size=(subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2),
                                         strides=(subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2),
                                         padding='same')
        self.ResNet_block_D_branch_end = ResNet_class(filters=round(convolutional_filters / subcarrier_strides_l2),
                                                      kernel_size=kernels_start_and_end,
                                                      strides=convolutional_strides,
                                                      dilation=convolutional_dilation,
                                                      trainable=trainablity)
        self.Tconv_D_2 = Conv3DTranspose(filters=2,
                                         kernel_size=(subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1),
                                         strides=(subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1),
                                         padding='same')
        self.MP_D_branch_end = MaxPooling3D(pool_size=(1, int(N_u_a / N_b_rf), int(N_b_a / N_s)))
        self.reshaper_D = Reshape(target_shape=[K, N_b_rf, N_s, 2])

        # V_RF path
        self.ResNet_block_RF_branch = []
        for i in range(self.n_D_and_RF_layers_specialized):
            self.ResNet_block_RF_branch.append(ResNet_class(filters=convolutional_filters,
                                                            kernel_size=kernels,
                                                            strides=convolutional_strides,
                                                            dilation=convolutional_dilation,
                                                            trainable=trainablity))

        self.Tconv_V_RF_1 = Conv3DTranspose(filters=convolutional_filters,
                                            kernel_size=(subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2),
                                            strides=(subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2),
                                            padding='same')
        self.ResNet_block_RF_branch_end = ResNet_class(filters=round(convolutional_filters / subcarrier_strides_l2),
                                                       kernel_size=kernels_start_and_end,
                                                       strides=convolutional_strides,
                                                       dilation=convolutional_dilation,
                                                       trainable=trainablity)
        self.Tconv_V_RF_2 = Conv3DTranspose(filters=1,
                                            kernel_size=(subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1),
                                            strides=(subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1),
                                            padding='same')
        self.MP_RF_branch_end = MaxPooling3D(pool_size=(K, int(N_u_a / N_b_a), N_b_a))

    def call(self, vd, vrf):

        # V_D path
        for i in range(0, self.n_D_and_RF_layers_specialized, 1):
            vd = self.ResNet_block_D_branch[i](vd)

        vd = self.Tconv_D_1(vd)
        vd = self.ResNet_block_D_branch_end(vd)
        vd = self.Tconv_D_2(vd)
        vd = self.MP_D_branch_end(vd)
        vd = self.reshaper_D(vd)

        # V_RF path
        for i in range(0, self.n_D_and_RF_layers_specialized, 1):
            vrf = self.ResNet_block_RF_branch[i](vrf)

        vrf = self.Tconv_V_RF_1(vrf)
        vrf = self.ResNet_block_RF_branch_end(vrf)
        vrf = self.Tconv_V_RF_2(vrf)
        vrf = self.MP_RF_branch_end(vrf)
        return vd, vrf


class CNN_model_class():
    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, SNR, P, N_c, N_scatterers,
                 angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, mat_fname,
                 dataset_size, dropout_rate, convolutional_kernels, convolutional_filters,
                 convolutional_strides, convolutional_dilation,
                 subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1,
                 subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2,
                 generic_part_trainable, specialized_part_trainable, n_common_layers, n_D_and_RF_layers_generic,
                 n_D_and_RF_layers_specialized):
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
        self.subcarrier_strides_l1 = subcarrier_strides_l1
        self.N_b_a_strides_l1 = N_b_a_strides_l1
        self.N_u_a_strides_l1 = N_u_a_strides_l1
        self.subcarrier_strides_l2 = subcarrier_strides_l2
        self.N_b_a_strides_l2 = N_b_a_strides_l2
        self.N_u_a_strides_l2 = N_u_a_strides_l2
        self.generic_part_trainable = generic_part_trainable
        self.specialized_part_trainable = specialized_part_trainable
        self.n_common_layers = n_common_layers
        self.n_D_and_RF_layers_generic = n_D_and_RF_layers_generic
        self.n_D_and_RF_layers_specialized = n_D_and_RF_layers_specialized

    def CNN_transceiver_large_sys(self, layer_name):
        kernels = [min(self.K, self.convolutional_kernels),
                   min(self.N_u_a, self.convolutional_kernels),
                   min(self.N_b_a, self.convolutional_kernels)]
        csi = Input(shape=(self.K, self.N_u_a, self.N_b_a, 2), batch_size=self.BATCHSIZE)
        kernels_start_and_end = [min(self.K, self.convolutional_kernels) + 2,
                                 min(self.N_u_a, self.convolutional_kernels) + 2,
                                 min(self.N_b_a, self.convolutional_kernels) + 2]
        generic_part_obj = Generic_2_ns_4_large_MIMOFDM_CNN_class(
            convolutional_filters=self.convolutional_filters,
            kernels=kernels,
            kernels_start_and_end=kernels_start_and_end,
            convolutional_strides=self.convolutional_strides,
            convolutional_dilation=self.convolutional_dilation,
            trainablity=self.generic_part_trainable,
            subcarrier_strides_l1=self.subcarrier_strides_l1,
            N_b_a_strides_l1=self.N_b_a_strides_l1,
            N_u_a_strides_l1=self.N_u_a_strides_l1,
            subcarrier_strides_l2=self.subcarrier_strides_l2,
            N_b_a_strides_l2=self.N_b_a_strides_l2,
            N_u_a_strides_l2=self.N_u_a_strides_l2,
            n_common_layers=self.n_common_layers,
            n_D_and_RF_layers_generic=self.n_D_and_RF_layers_generic,
            layer_name=layer_name + '_generic')

        specialized_part_obj = Specialized_2_ns_4_large_MIMOFDM_CNN_class(
            convolutional_filters=self.convolutional_filters,
            kernels=kernels,
            kernels_start_and_end=kernels_start_and_end,
            convolutional_strides=self.convolutional_strides,
            convolutional_dilation=self.convolutional_dilation,
            N_b_a=self.N_b_a,
            N_b_rf=self.N_b_rf,
            N_u_a=self.N_u_a,
            N_u_rf=self.N_u_rf,
            N_s=self.N_s,
            K=self.K,
            PTRS_seperation=self.PTRS_seperation,
            trainablity=self.specialized_part_trainable,
            subcarrier_strides_l1=self.subcarrier_strides_l1,
            N_b_a_strides_l1=self.N_b_a_strides_l1,
            N_u_a_strides_l1=self.N_u_a_strides_l1,
            subcarrier_strides_l2=self.subcarrier_strides_l2,
            N_b_a_strides_l2=self.N_b_a_strides_l2,
            N_u_a_strides_l2=self.N_u_a_strides_l2,
            n_D_and_RF_layers_specialized=self.n_D_and_RF_layers_specialized,
            layer_name=layer_name + '_specialized')

        # Models
        vd_generic, vrf_generic = generic_part_obj(csi)
        vd, vrf = specialized_part_obj(vd_generic, vrf_generic)
        return Model(inputs=[csi], outputs=[vd, vrf])

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
