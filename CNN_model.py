import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv3D, Reshape, MaxPooling3D, BatchNormalization, Input, Flatten, Dropout


class CNN_model_class:
    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, mat_fname,
                 dataset_size, width_parameter, dropout_rate):
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
        self.Ts = Ts
        self.fc = fc
        self.c = c
        self.mat_fname = mat_fname
        self.dataset_size = dataset_size
        self.width_parameter = width_parameter
        self.dropout_rate = dropout_rate

    def custom_CNN_plus_FC_with_functional_API(self):
        csi = Input(shape=(self.K, self.N_u_a, self.N_b_a, 2), batch_size=self.BATCHSIZE)

        # CONV layers
        C1 = Conv3D(filters=int(16 * self.width_parameter + 1), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                    padding='same',
                    activation='relu')
        MP1 = MaxPooling3D(pool_size=(2, 1, 1), padding='same')
        BN1 = BatchNormalization()
        C2 = Conv3D(filters=int(32 * self.width_parameter + 1), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                    padding='same',
                    activation='relu')
        MP2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')
        BN2 = BatchNormalization()

        # FC layers
        Layer_flatten = Flatten()
        FC1 = Dense(int(self.K * self.N_u_a * self.N_b_a * 2 * self.width_parameter + 1), activation='relu')
        FC2 = Dense(int(self.K * self.N_u_a * self.N_b_a * 2 * self.width_parameter + 1), activation='relu')

        # specifics
        Layer_BN8 = BatchNormalization()
        Layer_BN9 = BatchNormalization()
        Layer_BN10 = BatchNormalization()
        Layer_BN11 = BatchNormalization()
        Layer_BN12 = BatchNormalization()
        Layer_BN13 = BatchNormalization()
        Layer_BN14 = BatchNormalization()
        Layer_BN15 = BatchNormalization()
        Layer_BN16 = BatchNormalization()
        Layer_BN17 = BatchNormalization()
        Layer_BN18 = BatchNormalization()
        Layer_BN19 = BatchNormalization()
        Layer_BN20 = BatchNormalization()
        Layer_BN21 = BatchNormalization()
        Layer_BN22 = BatchNormalization()
        Layer_BN23 = BatchNormalization()
        Layer_BN24 = BatchNormalization()
        Layer_BN25 = BatchNormalization()
        Layer_BN26 = BatchNormalization()
        Layer_BN27 = BatchNormalization()
        Layer_BN28 = BatchNormalization()
        Layer_BN29 = BatchNormalization()
        Layer_BN30 = BatchNormalization()
        Layer_BN31 = BatchNormalization()

        Layer_drop1 = Dropout(self.dropout_rate)
        Layer_drop2 = Dropout(self.dropout_rate)
        Layer_drop3 = Dropout(self.dropout_rate)
        Layer_drop4 = Dropout(self.dropout_rate)
        Layer_drop5 = Dropout(self.dropout_rate)
        Layer_drop6 = Dropout(self.dropout_rate)
        Layer_drop7 = Dropout(self.dropout_rate)
        Layer_drop8 = Dropout(self.dropout_rate)
        Layer_drop9 = Dropout(self.dropout_rate)
        Layer_drop10 = Dropout(self.dropout_rate)
        Layer_drop11 = Dropout(self.dropout_rate)
        Layer_drop12 = Dropout(self.dropout_rate)

        Layer_V_D_1 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                            activation='relu')
        Layer_V_D_2 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                            activation='relu')
        Layer_V_D_3 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                            activation='relu')
        Layer_V_D_4 = Dense(int(self.K * self.N_b_rf * self.N_s * 2 * self.width_parameter + 1), # size B
                            activation='relu')
        Layer_V_D_5 = Dense(int(self.K * self.N_b_rf * self.N_s * 2 * self.width_parameter + 1), # size B
                            activation='relu')
        Layer_V_D_6 = Dense(int(self.K * self.N_b_rf * self.N_s * 2 * self.width_parameter + 1), # size B
                            activation='relu')
        Layer_V_D_7 = Dense(self.K * self.N_b_rf * self.N_s * 2)
        reshaper_V_D = Reshape(target_shape=[self.K, self.N_b_rf, self.N_s, 2])


        Layer_W_D_1 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1), # size A
                            activation='relu')
        Layer_W_D_2 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1), # size A
                            activation='relu')
        Layer_W_D_3 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1),  # size A
                            activation='relu')
        Layer_W_D_4 = Dense(int(self.K * self.N_u_rf * self.N_s * 2*  self.width_parameter + 1), # size B
                            activation='relu')
        Layer_W_D_5 = Dense(int(self.K * self.N_u_rf * self.N_s * 2*  self.width_parameter + 1), # size B
                            activation='relu')
        Layer_W_D_6 = Dense(int(self.K * self.N_u_rf * self.N_s * 2*  self.width_parameter + 1), # size B
                            activation='relu')
        Layer_W_D_7 = Dense(self.K * self.N_u_rf * self.N_s * 2, activation='relu')
        reshaper_W_D = Reshape(target_shape=[self.K, self.N_u_rf, self.N_s, 2])


        Layer_V_RF_1 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_V_RF_2 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_V_RF_3 = Dense(int(self.K * self.N_b_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_V_RF_4 = Dense(int(self.N_b_a * 2 * self.width_parameter + 1), # size B
                             activation='relu')
        Layer_V_RF_5 = Dense(int(self.N_b_a * self.width_parameter + 1), # size C
                             activation='relu')
        Layer_V_RF_6 = Dense(int(self.N_b_a * self.width_parameter + 1), # size C
                             activation='relu')
        Layer_V_RF_7 = Dense(self.N_b_a, activation='relu')
        # reshaper_V_RF = Reshape(target_shape=[self.N_b_a])

        Layer_W_RF_1 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_W_RF_2 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_W_RF_3 = Dense(int(self.K * self.N_u_a * 2 * self.width_parameter + 1), # size A
                             activation='relu')
        Layer_W_RF_4 = Dense(int(self.N_u_a * 2 * self.width_parameter + 1), # size B
                             activation='relu')
        Layer_W_RF_5 = Dense(int(self.N_u_a * self.width_parameter + 1), # size C
                             activation='relu')
        Layer_W_RF_6 = Dense(int(self.N_u_a * self.width_parameter + 1), # size C
                             activation='relu')
        Layer_W_RF_7 = Dense(self.N_u_a, activation='relu')
        # reshaper_W_RF = Reshape(target_shape=[self.N_u_a])

        # Connections
        x = C1(csi)
        # x = MP1(x)
        # x = BN1(x)
        # x = C2(x)
        # x = MP2(x)
        x = BN2(x)
        x = Layer_flatten(x)
        x = FC1(x)
        x = FC2(x)

        vd = Layer_V_D_1(x)
        vd = Layer_BN8(vd)
        vd = Layer_V_D_2(vd)
        vd = Layer_BN9(vd)
        vd = Layer_drop1(vd)
        vd = Layer_V_D_3(vd)
        vd = Layer_BN10(vd)
        vd = Layer_V_D_4(vd)
        vd = Layer_BN11(vd)
        vd = Layer_V_D_5(vd)
        vd = Layer_BN12(vd)
        vd = Layer_drop2(vd)
        vd = Layer_V_D_6(vd)
        vd = Layer_BN13(vd)
        vd = Layer_V_D_7(vd)
        vd = reshaper_V_D(vd)

        wd = Layer_W_D_1(x)
        wd = Layer_BN14(wd)
        wd = Layer_W_D_2(wd)
        wd = Layer_BN15(wd)
        wd = Layer_drop3(wd)
        wd = Layer_W_D_3(wd)
        wd = Layer_BN16(wd)
        wd = Layer_W_D_4(wd)
        wd = Layer_BN17(wd)
        wd = Layer_W_D_5(wd)
        wd = Layer_BN18(wd)
        wd = Layer_drop4(wd)
        wd = Layer_W_D_6(wd)
        wd = Layer_BN19(wd)
        wd = Layer_W_D_7(wd)
        wd = reshaper_W_D(wd)

        vrf = Layer_V_RF_1(x)
        vrf = Layer_BN20(vrf)
        vrf = Layer_V_RF_2(vrf)
        vrf = Layer_BN21(vrf)
        vrf = Layer_drop5(vrf)
        vrf = Layer_V_RF_3(vrf)
        vrf = Layer_BN22(vrf)
        vrf = Layer_V_RF_4(vrf)
        vrf = Layer_BN23(vrf)
        vrf = Layer_V_RF_5(vrf)
        vrf = Layer_BN24(vrf)
        vrf = Layer_drop6(vrf)
        vrf = Layer_V_RF_6(vrf)
        vrf = Layer_BN25(vrf)
        vrf = Layer_V_RF_7(vrf)
        # vrf = reshaper_V_RF(vrf)

        wrf = Layer_W_RF_1(x)
        wrf = Layer_BN26(wrf)
        wrf = Layer_W_RF_2(wrf)
        wrf = Layer_BN27(wrf)
        wrf = Layer_drop7(wrf)
        wrf = Layer_W_RF_3(wrf)
        wrf = Layer_BN28(wrf)
        wrf = Layer_W_RF_4(wrf)
        wrf = Layer_BN29(wrf)
        wrf = Layer_W_RF_5(wrf)
        wrf = Layer_BN30(wrf)
        wrf = Layer_drop8(wrf)
        wrf = Layer_W_RF_6(wrf)
        wrf = Layer_BN31(wrf)
        wrf = Layer_W_RF_7(wrf)
        # wrf = reshaper_W_RF(wrf)

        func_model = Model(inputs=csi, outputs=[vd, wd, vrf, wrf])
        return func_model



    # sequential implementation
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def custom_actication(self, inputs):
        V_D, W_D, vrf, wrf = inputs

        V_D_cplx = tf.complex(V_D[:, :, :, :, 0], V_D[:, :, :, :, 1])
        W_D_cplx = tf.complex(W_D[:, :, :, :, 0], W_D[:, :, :, :, 1])
        vrf_cplx = tf.complex(tf.cos(vrf), tf.sin(vrf))
        wrf_cplx = tf.complex(tf.cos(wrf), tf.sin(wrf))

        # partially-connected analog beamformer matrix implementation ----------------
        V_RF_list_forall_samples = []
        W_RF_list_forall_samples = []
        V_D_new_forall_samples = []

        for ij in range(self.BATCHSIZE):
            # partially-connected analog beamformer matrix implementation --------------

            # for BS
            vrf_zero_padded = tf.concat([tf.reshape(vrf_cplx[ij, :], shape=[self.N_b_a, 1]),
                                         tf.zeros(shape=[self.N_b_a, self.N_b_rf - 1], dtype=tf.complex64)], axis=1)
            # print('vrf_zero_padded', vrf_zero_padded.shape)
            r_bs = int(self.N_b_a / self.N_b_rf)
            T2_BS = []
            for i in range(self.N_b_rf):
                T0_BS = vrf_zero_padded[r_bs * i: r_bs * (i + 1), :]
                # print('T0_BS', T0_BS.shape)
                T1_BS = tf.roll(T0_BS, shift=i, axis=1)
                # print('T1_BS', T1_BS.shape)
                T2_BS.append(T1_BS)
            V_RF_per_sample = tf.concat(T2_BS, axis=0)
            # print('V_RF_per_sample', V_RF_per_sample.shape)
            V_RF_list_forall_samples.append(V_RF_per_sample)

            # for UE
            wrf_zero_padded = tf.concat([tf.reshape(wrf_cplx[ij, :], shape=[self.N_u_a, 1]),
                                         tf.zeros(shape=[self.N_u_a, self.N_u_rf - 1], dtype=tf.complex64)], axis=1)
            r_ue = int(self.N_u_a / self.N_u_rf)
            T2_UE = []
            for i in range(self.N_u_rf):
                T0_UE = wrf_zero_padded[r_ue * i: r_ue * (i + 1), :]
                T1_UE = tf.roll(T0_UE, shift=i, axis=1)
                T2_UE.append(T1_UE)
            W_RF_per_sample = tf.concat(T2_UE, axis=0)
            W_RF_list_forall_samples.append(W_RF_per_sample)

            # per subcarrier power normalization ---------------------------------------
            V_D_new_per_sample = []
            # denum = tf.zeros( shape=[1] , dtype=tf.complex64)
            for k in range(self.K):
                T0 = tf.linalg.matmul(V_RF_per_sample, V_D_cplx[ij, k, :, :], adjoint_a=False, adjoint_b=False)
                T1 = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
                # denum = tf.add(denum , tf.linalg.trace(T1))
                denum = (tf.linalg.trace(T1))#, tf.complex(1e-16,
                                                    #           1e-16))  ####################################################### numeric precision flaw
                # denum = tf.linalg.trace(T1)
                # V_D_new_forall_samples.append(tf.divide( tf.multiply(V_D_cplx[ij,:,:,:] , tf.cast(tf.sqrt(P) ,dtype=tf.complex64)) , tf.sqrt(denum)))
                V_D_new_per_sample.append(
                    tf.divide(tf.multiply(V_D_cplx[ij, k, :, :], tf.cast(tf.sqrt(self.P), dtype=tf.complex64)),
                              tf.sqrt(denum)))
            V_D_new_forall_samples.append(tf.stack(V_D_new_per_sample, axis=0))

        V_RF_cplx = tf.stack(V_RF_list_forall_samples, axis=0)
        W_RF_cplx = tf.stack(W_RF_list_forall_samples, axis=0)
        V_D_new = tf.stack(V_D_new_forall_samples, axis=0)

        # print(V_RF_cplx.shape)

        return V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx



    # # map_fn implementation
    # @tf.function
    # @tf.autograph.experimental.do_not_convert
    # def custom_actication(self, inputs):
    #     V_D, W_D, vrf, wrf = inputs
    #
    #     V_D_cplx = tf.complex(V_D[:, :, :, :, 0], V_D[:, :, :, :, 1])
    #     W_D_cplx = tf.complex(W_D[:, :, :, :, 0], W_D[:, :, :, :, 1])
    #     vrf_cplx = tf.complex(tf.cos(vrf), tf.sin(vrf))
    #     wrf_cplx = tf.complex(tf.cos(wrf), tf.sin(wrf))
    #
    #     # partially-connected analog beamformer matrix implementation ----------------
    #
    #     # --------------------------------------------------------------------------------------------------------------
    #     bundeled_inputs_0 = [vrf_cplx, wrf_cplx, V_D_cplx]
    #     # print('vrf_cplx shape', vrf_cplx.shape)
    #     # print('wrf_cplx shape', wrf_cplx.shape)
    #     # print('V_D_cplx shape', V_D_cplx.shape)
    #
    #     V_RF_cplx, W_RF_cplx, V_D_new_cplx = tf.map_fn(self.custorm_activation_per_sample, bundeled_inputs_0,
    #                                                fn_output_signature=(tf.complex64, tf.complex64, tf.complex64), parallel_iterations=self.BATCHSIZE)
    #
    #     return V_D_new_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx
    #
    # @tf.function
    # def normalize_power_per_subcarrier(self, bundeled_inputs_0):
    #     V_RF, V_D_k = bundeled_inputs_0
    #     T0 = tf.linalg.matmul(V_RF, V_D_k, adjoint_a=False, adjoint_b=False)
    #     T1 = tf.linalg.matmul(T0, T0, adjoint_a=False, adjoint_b=True)
    #     # denum = tf.add(denum , tf.linalg.trace(T1))
    #     denum = tf.add(tf.linalg.trace(T1), tf.complex(1e-16, 1e-16)) ###### numeric precision flaw
    #     V_D_k_normalized = tf.divide(tf.multiply(V_D_k, tf.cast(tf.sqrt(self.P), dtype=tf.complex64)),tf.sqrt(denum))
    #     return V_D_k_normalized
    #
    #
    # @tf.function
    # @tf.autograph.experimental.do_not_convert
    # def custorm_activation_per_sample(self, bundeled_inputs_0):
    #     vrf_cplx, wrf_cplx, V_D_cplx = bundeled_inputs_0
    #     # for BS
    #     vrf_zero_padded = tf.concat([tf.reshape(vrf_cplx, shape=[self.N_b_a, 1]),
    #                                  tf.zeros(shape=[self.N_b_a, self.N_b_rf - 1], dtype=tf.complex64)], axis=1)
    #     # print('vrf_zero_padded', vrf_zero_padded.shape)
    #     r_bs = int(self.N_b_a / self.N_b_rf)
    #     T2_BS = []
    #     for i in range(self.N_b_rf):
    #         T0_BS = vrf_zero_padded[r_bs * i: r_bs * (i + 1), :]
    #         # print('T0_BS', T0_BS.shape)
    #         T1_BS = tf.roll(T0_BS, shift=i, axis=1)
    #         # print('T1_BS', T1_BS.shape)
    #         T2_BS.append(T1_BS)
    #     V_RF_per_sample = tf.concat(T2_BS, axis=0)
    #
    #     # for UE
    #     wrf_zero_padded = tf.concat([tf.reshape(wrf_cplx, shape=[self.N_u_a, 1]),
    #                                  tf.zeros(shape=[self.N_u_a, self.N_u_rf - 1], dtype=tf.complex64)], axis=1)
    #     r_ue = int(self.N_u_a / self.N_u_rf)
    #     T2_UE = []
    #     for i in range(self.N_u_rf):
    #         T0_UE = wrf_zero_padded[r_ue * i: r_ue * (i + 1), :]
    #         T1_UE = tf.roll(T0_UE, shift=i, axis=1)
    #         T2_UE.append(T1_UE)
    #     W_RF_per_sample = tf.concat(T2_UE, axis=0)
    #
    #     # per subcarrier power normalization ---------------------------------------
    #
    #     # repeating inputs for vectorization
    #     V_RF_per_sample_repeated_K_times = tf.tile([V_RF_per_sample], multiples=[self.K, 1, 1])
    #     bundeled_inputs_1 = [V_RF_per_sample_repeated_K_times, V_D_cplx]
    #     V_D_cplx_normalized_per_sample = tf.map_fn(self.normalize_power_per_subcarrier, bundeled_inputs_1,
    #                                     fn_output_signature=tf.complex64, parallel_iterations=self.K)
    #
    #
    #     return V_RF_per_sample, W_RF_per_sample, V_D_cplx_normalized_per_sample
