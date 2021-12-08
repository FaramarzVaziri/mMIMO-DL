import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, Reshape, MaxPooling3D, BatchNormalization, Input, Flatten, Dropout, \
    Activation, Add, Conv3DTranspose, Multiply, Lambda, Embedding, Concatenate, AveragePooling3D


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


class ns_adaptive_module_class(tf.keras.Model):  #
    def __init__(self, n1, n2, n3, n_chan, trainable_ns):
        super(ns_adaptive_module_class, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n_chan = n_chan


        self.bn_ns = BatchNormalization(trainable=trainable_ns)
        self.AP = AveragePooling3D(pool_size=(n1, n2, n3))
        self.bn_common_main_data = BatchNormalization(trainable=trainable_ns)

        self.FC_mul_1 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.FC_mul_2 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.bn_mul = BatchNormalization(trainable=trainable_ns)
        self.FC_mul_3 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.FC_mul_4 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)

        self.FC_add_1 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.FC_add_2 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.bn_add = BatchNormalization(trainable=trainable_ns)
        self.FC_add_3 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)
        self.FC_add_4 = Dense(units=n_chan, activation='relu', use_bias=True, trainable=trainable_ns)

        self.cat = Concatenate(axis=1)

    def call(self, input_tensor, ns):
        y = self.AP(input_tensor)
        # y = self.bn_common_main_data(y)
        # ns = self.bn_ns(ns)

        y = self.cat([ns, tf.squeeze(y)])

        m = self.FC_mul_1(y)
        m = self.FC_mul_2(m)
        # m = self.bn_mul(m)
        # m = self.FC_mul_3(m)
        # m = self.FC_mul_4(m)
        m = tf.tile(tf.reshape(m, shape=[-1, 1, 1, 1, self.n_chan]), multiples=[1, self.n1, self.n2, self.n3, 1])

        b = self.FC_add_1(y)
        b = self.FC_add_2(b)
        # b = self.bn_add(b)
        # b = self.FC_add_3(b)
        # b = self.FC_add_4(b)
        b = tf.tile(tf.reshape(b, shape=[-1, 1, 1, 1, self.n_chan]), multiples=[1, self.n1, self.n2, self.n3, 1])

        z = tf.multiply(input_tensor, m)
        z = tf.add(z, b)
        return z


class CNN_model_class():
    def __init__(self, P, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, BATCHSIZE,
                 truncation_ratio_keep, Nsymb, dropout_rate, convolutional_kernels, convolutional_filters,
                 convolutional_strides, convolutional_dilation,
                 subcarrier_strides_l1, N_u_a_strides_l1, N_b_a_strides_l1,
                 subcarrier_strides_l2, N_u_a_strides_l2, N_b_a_strides_l2, n_common_layers, n_D_and_RF_layers,
                 extra_kernel):
        self.P = P
        self.N_b_a = N_b_a
        self.N_b_rf = N_b_rf
        self.N_u_a = N_u_a
        self.N_u_rf = N_u_rf
        self.N_s = N_s
        self.K = K
        self.PTRS_seperation = PTRS_seperation
        self.BATCHSIZE = BATCHSIZE
        self.truncation_ratio_keep = truncation_ratio_keep
        self.Nsymb = Nsymb
        self.dropout_rate = dropout_rate
        self.convolutional_kernels = convolutional_kernels
        self.convolutional_filters = convolutional_filters
        self.convolutional_strides = convolutional_strides
        self.convolutional_dilation = convolutional_dilation
        self.subcarrier_strides_l1 = subcarrier_strides_l1
        self.N_u_a_strides_l1 = N_u_a_strides_l1
        self.N_b_a_strides_l1 = N_b_a_strides_l1
        self.subcarrier_strides_l2 = subcarrier_strides_l2
        self.N_u_a_strides_l2 = N_u_a_strides_l2
        self.N_b_a_strides_l2 = N_b_a_strides_l2
        self.n_common_layers = n_common_layers
        self.n_D_and_RF_layers = n_D_and_RF_layers
        self.extra_kernel = extra_kernel

    def CNN_transceiver(self, trainable_csi, trainable_ns, layer_name):
        kernels_start_and_end = [min(self.K, self.convolutional_kernels) + self.extra_kernel,
                                 min(self.N_u_a, self.convolutional_kernels) + self.extra_kernel,
                                 min(self.N_b_a, self.convolutional_kernels) + self.extra_kernel]
        kernels = [min(self.K, self.convolutional_kernels),
                   min(self.N_u_a, self.convolutional_kernels),
                   min(self.N_b_a, self.convolutional_kernels)]
        csi = Input(shape=(self.K, self.N_u_a, self.N_b_a, 2), batch_size=self.BATCHSIZE)
        ns = Input(shape=[1], batch_size=self.BATCHSIZE)

        # common path
        AP_1 = AveragePooling3D(pool_size=(self.subcarrier_strides_l1, self.N_u_a_strides_l1, self.N_b_a_strides_l1))
        AP_2 = AveragePooling3D(pool_size=(self.subcarrier_strides_l2, self.N_u_a_strides_l2, self.N_b_a_strides_l2))

        ResNet_block_common_branch = []
        ResNet_block_common_branch.append(ResNet_class(
            filters=round(self.convolutional_filters / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
            kernel_size=kernels_start_and_end,
            strides=self.convolutional_strides,
            dilation=self.convolutional_dilation,
            trainable=trainable_csi))
        ResNet_block_common_branch.append(
            ResNet_class(filters=round(self.convolutional_filters / (self.subcarrier_strides_l2)),
                         kernel_size=kernels,
                         strides=self.convolutional_strides,
                         dilation=self.convolutional_dilation,
                         trainable=trainable_csi))
        for i in range(2, self.n_common_layers, 1):
            ResNet_block_common_branch.append(ResNet_class(filters=self.convolutional_filters,
                                                           kernel_size=kernels,
                                                           strides=self.convolutional_strides,
                                                           dilation=self.convolutional_dilation,
                                                           trainable=trainable_csi))

        ns_adaptive_module_c = ns_adaptive_module_class(
            n1=round(self.K / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
            n2=round(self.N_b_a / (self.N_b_a_strides_l1 * self.N_b_a_strides_l2)),
            n3=round(self.N_u_a / (self.N_u_a_strides_l1 * self.N_u_a_strides_l2)),
            n_chan=self.convolutional_filters, trainable_ns=trainable_ns)

        # V_D path
        ResNet_block_D_branch = []
        for i in range(self.n_D_and_RF_layers):
            ResNet_block_D_branch.append(ResNet_class(filters=self.convolutional_filters,
                                                      kernel_size=kernels,
                                                      strides=self.convolutional_strides,
                                                      dilation=self.convolutional_dilation,
                                                      trainable=trainable_csi))

        ns_adaptive_module_D = ns_adaptive_module_class(
            n1=round(self.K / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
            n2=round(self.N_b_a / (self.N_b_a_strides_l1 * self.N_b_a_strides_l2)),
            n3=round(self.N_u_a / (self.N_u_a_strides_l1 * self.N_u_a_strides_l2)),
            n_chan=self.convolutional_filters, trainable_ns=trainable_ns)

        Tconv_D_1 = Conv3DTranspose(filters=self.convolutional_filters,
                                    kernel_size=[self.subcarrier_strides_l2, self.N_u_a_strides_l2,
                                                 self.N_b_a_strides_l2],
                                    strides=(self.subcarrier_strides_l2, self.N_u_a_strides_l2, self.N_b_a_strides_l2),
                                    padding='same')
        # ns_adaptive_module_d = ns_adaptive_module_class(
        #     n1=round(self.K / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
        #     n2=round(self.N_b_a / (self.N_b_a_strides_l1 * self.N_b_a_strides_l2)),
        #     n3=round(self.N_u_a / (self.N_u_a_strides_l1 * self.N_u_a_strides_l2)),
        #     n_chan=self.convolutional_filters)
        ResNet_block_D_branch_end = ResNet_class(filters=round(self.convolutional_filters / self.subcarrier_strides_l2),
                                                 kernel_size=kernels_start_and_end,
                                                 strides=self.convolutional_strides,
                                                 dilation=self.convolutional_dilation,
                                                 trainable=trainable_csi)
        Tconv_D_2 = Conv3DTranspose(filters=2,
                                    kernel_size=[self.subcarrier_strides_l1, self.N_u_a_strides_l1,
                                                 self.N_b_a_strides_l1],
                                    strides=[self.subcarrier_strides_l1, self.N_u_a_strides_l1, self.N_b_a_strides_l1],
                                    padding='same')
        MP_D_branch_end = MaxPooling3D(pool_size=(1, int(self.N_u_a / self.N_b_rf), int(self.N_b_a / self.N_s)))
        reshaper_D = Reshape(target_shape=[self.K, self.N_b_rf, self.N_s, 2])

        # V_RF path
        ResNet_block_RF_branch = []
        for i in range(self.n_D_and_RF_layers):
            ResNet_block_RF_branch.append(ResNet_class(filters=self.convolutional_filters,
                                                       kernel_size=kernels,
                                                       strides=self.convolutional_strides,
                                                       dilation=self.convolutional_dilation,
                                                       trainable=trainable_csi))

        ns_adaptive_module_RF = ns_adaptive_module_class(
            n1=round(self.K / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
            n2=round(self.N_b_a / (self.N_b_a_strides_l1 * self.N_b_a_strides_l2)),
            n3=round(self.N_u_a / (self.N_u_a_strides_l1 * self.N_u_a_strides_l2)),
            n_chan=self.convolutional_filters, trainable_ns=trainable_ns)

        Tconv_V_RF_1 = Conv3DTranspose(filters=self.convolutional_filters,
                                       kernel_size=[self.subcarrier_strides_l2, self.N_u_a_strides_l2,
                                                    self.N_b_a_strides_l2],
                                       strides=(
                                       self.subcarrier_strides_l2, self.N_u_a_strides_l2, self.N_b_a_strides_l2),
                                       padding='same')
        # ns_adaptive_module_rf = ns_adaptive_module_class(
        #     n1=round(self.K / (self.subcarrier_strides_l1 * self.subcarrier_strides_l2)),
        #     n2=round(self.N_b_a / (self.N_b_a_strides_l1 * self.N_b_a_strides_l2)),
        #     n3=round(self.N_u_a / (self.N_u_a_strides_l1 * self.N_u_a_strides_l2)),
        #     n_chan=self.convolutional_filters)

        ResNet_block_RF_branch_end = ResNet_class(
            filters=round(self.convolutional_filters / self.subcarrier_strides_l1),
            kernel_size=kernels_start_and_end,
            strides=self.convolutional_strides,
            dilation=self.convolutional_dilation,
            trainable=trainable_csi)
        Tconv_V_RF_2 = Conv3DTranspose(filters=1,
                                       kernel_size=[self.subcarrier_strides_l1, self.N_u_a_strides_l1,
                                                    self.N_b_a_strides_l1],
                                       strides=[self.subcarrier_strides_l1, self.N_u_a_strides_l1,
                                                self.N_b_a_strides_l1],
                                       padding='same')
        MP_RF_branch_end = MaxPooling3D(pool_size=(self.K, int(self.N_u_a / self.N_b_a), self.N_b_a))

        # connections
        x = ResNet_block_common_branch[0](csi)
        x = AP_1(x)
        x = ResNet_block_common_branch[1](x)
        x = AP_2(x)
        for i in range(2, self.n_common_layers, 1):
            x = ResNet_block_common_branch[i](x)

        x = ns_adaptive_module_c(x, ns)

        # V_D path
        vd = ResNet_block_D_branch[0](x)
        for i in range(1, self.n_D_and_RF_layers, 1):
            vd = ResNet_block_D_branch[i](vd)

        vd = ns_adaptive_module_D(vd, ns)

        vd = Tconv_D_1(vd)
        vd = ResNet_block_D_branch_end(vd)
        vd = Tconv_D_2(vd)
        vd = MP_D_branch_end(vd)
        vd = reshaper_D(vd)

        # V_RF path
        vrf = ResNet_block_RF_branch[0](x)
        for i in range(1, self.n_D_and_RF_layers, 1):
            vrf = ResNet_block_RF_branch[i](vrf)

        vrf = ns_adaptive_module_RF(vrf, ns)

        vrf = Tconv_V_RF_1(vrf)
        vrf = ResNet_block_RF_branch_end(vrf)
        vrf = Tconv_V_RF_2(vrf)
        vrf = MP_RF_branch_end(vrf)

        func_model = Model(inputs=[csi, ns], outputs=[vd, vrf], name=layer_name)
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
