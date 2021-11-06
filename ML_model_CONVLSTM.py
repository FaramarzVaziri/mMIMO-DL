import tensorflow as tf
import numpy as np
import scipy.io as sio
import tensorflow as tf
from dataset_generator import dataset_generator_class
global ofdm_symbol_id

loss_metric = tf.keras.metrics.Mean(name='neg_capacity')
loss_metric_test = tf.keras.metrics.Mean(name='neg_capacity_test')
norm_records = tf.keras.metrics.Mean(name='norm')
capacity_metric_test = tf.keras.metrics.Mean(name='neg_capacity_performance_metric')


class ML_model_class(tf.keras.Model):

    def __init__(self, CNN_transmitter, CNN_receiver, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, PTRS_seperation, SNR, P,
                 N_c, N_scatterers, angular_spread_rad, wavelength, d, BATCHSIZE, phase_shift_stddiv,
                 truncation_ratio_keep, sampling_ratio_time_domain_keep, Nsymb, Ts, fc, c, PHN_innovation_std,
                 mat_fname, eval_dataset_size, mode,
                 phase_noise_exists_while_training, number_of_OFDM_symbols_considered):
        super(ML_model_class, self).__init__()
        self.CNN_transmitter = CNN_transmitter
        self.CNN_receiver = CNN_receiver
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
        self.sampling_ratio_time_domain_keep = sampling_ratio_time_domain_keep
        self.Nsymb = Nsymb
        self.Ts = Ts
        self.fc = fc
        self.c = c
        self.mat_fname = mat_fname
        self.eval_dataset_size = eval_dataset_size
        self.PHN_innovation_std = PHN_innovation_std
        self.mode = mode
        self.phase_noise_exists_while_training = phase_noise_exists_while_training
        self.number_of_OFDM_symbols_considered = number_of_OFDM_symbols_considered

    def compile(self, optimizer, loss, activation_TX, activation_RX,
                metric_capacity_in_presence_of_phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.metric_capacity_in_presence_of_phase_noise = metric_capacity_in_presence_of_phase_noise
        self.activation_TX = activation_TX
        self.activation_RX = activation_RX

    @tf.function
    def NN_input_preparation(self, H_tilde):
        csi_tx = tf.tile(tf.expand_dims(H_tilde[:, 0, :, :, :, :], axis=1),
                         multiples=[1, self.Nsymb, 1, 1, 1, 1])
        subcarrier_mask_ptrs = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(
            indices=tf.reshape(tf.cast(tf.range(0, self.K, self.PTRS_seperation), tf.int64),
                               shape=[round(self.K / self.PTRS_seperation), 1]),
            values=tf.ones(shape=[round(self.K / self.PTRS_seperation)], dtype=tf.int32),
            dense_shape=[self.K])))

        H_tilde_ptrs = tf.boolean_mask(H_tilde, subcarrier_mask_ptrs,
                                       axis=2)  # batch, Nsymb, K, ... [should mask on the dimension k]
        subcarrier_mask_non_ptrs = 1 - subcarrier_mask_ptrs
        H_tilde_non_ptrs = tf.boolean_mask(H_tilde, subcarrier_mask_non_ptrs,
                                           axis=2)  # batch, Nsymb, K, ... [should mask on the dimension k]

        # the following line produces the interleaved measured fresh H_tilde_ns for some k and old H_tilde_0 for the rest of the k
        uu= tf.tile(tf.expand_dims(H_tilde_non_ptrs[:, 0, :, :, :, :], axis=1),
                    multiples=[1, self.Nsymb, 1, 1, 1, 1])
        H_tilde_ptrs_and_non_ptrs_stacked = tf.concat([H_tilde_ptrs, uu], axis=2)
        csi_rx = H_tilde_ptrs_and_non_ptrs_stacked# tf.reshape(H_tilde_ptrs_and_non_ptrs_stacked,
                            # [self.BATCHSIZE, self.Nsymb, self.K, self.N_u_a, self.N_b_a, 2])
        return csi_tx, csi_rx

    @tf.function
    def train_step(self, inputs0):
        if (self.phase_noise_exists_while_training == False):
            _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U, set_of_ns = inputs0
            # 5 4          6         5

            with tf.GradientTape() as tape:

                V_D_tmp = []
                V_RF_tmp = []
                W_D_tmp = []
                W_RF_tmp = []

                rand_start = np.random.random_integers(low= 0, high= self.Nsymb - self.number_of_OFDM_symbols_considered)
                # selected_symbols = range(rand_start, rand_start + self.number_of_OFDM_symbols_considered)
                selected_symbols = range(self.Nsymb)

                V_D, V_RF = self.CNN_transmitter(tf.tile(tf.expand_dims(H_tilde[:, 0, :, :, :, :], axis=1),
                                                     multiples=[1, round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1, 1]))
                W_D, W_RF = self.CNN_receiver(tf.tile(tf.expand_dims(H_tilde[:, 0, :, :, :, :], axis=1),
                                                     multiples=[1, round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1, 1]))
                i=0
                for ns in selected_symbols:

                    VV_D, VV_RF = self.activation_TX([V_D[:,i,:], V_RF[:,i,:]])
                    V_D_tmp.append(VV_D)
                    V_RF_tmp.append(VV_RF)

                    WW_D, WW_RF = self.activation_RX([W_D[:,i,:], W_RF[:,i,:]])  # batch, 1, K, ...
                    W_D_tmp.append(WW_D)
                    W_RF_tmp.append(WW_RF)
                    i=i+1

                V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
                V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]

                W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
                W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]
                inputs2 = [V_D, W_D, tf.squeeze(H_tilde_complex[:,0,:,:,:]), V_RF, W_RF]
                d_loss = self.loss(inputs2)

            trainables = self.CNN_transmitter.trainable_weights + self.CNN_receiver.trainable_weights
            grads = tape.gradient(d_loss, trainables)
            self.optimizer.apply_gradients(zip(grads, trainables))
            loss_metric.update_state(d_loss)
            return {"neg_capacity_train_loss": loss_metric.result()}
        else:
            _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U, set_of_ns = inputs0
            # 5 4          6         5
            # rand_start = np.random.random_integers(low=0,
            #                                        high=self.Nsymb - self.number_of_OFDM_symbols_considered)
            # selected_symbols = range(rand_start, rand_start + self.number_of_OFDM_symbols_considered)
            selected_symbols = range(self.Nsymb)
            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

            with tf.GradientTape() as tape:

                V_D_tmp = []
                V_RF_tmp = []
                W_D_tmp = []
                W_RF_tmp = []
                V_D, V_RF = self.CNN_transmitter(csi_tx)
                W_D, W_RF = self.CNN_receiver(csi_rx)

                i=0
                for ns in selected_symbols:
                    VV_D, VV_RF = self.activation_TX([V_D[:,i,:], V_RF[:,i,:]])
                    V_D_tmp.append(VV_D)
                    V_RF_tmp.append(VV_RF)

                    WW_D, WW_RF = self.activation_RX([W_D[:,i,:], W_RF[:,i,:]])  # batch, 1, K, ...
                    W_D_tmp.append(WW_D)
                    W_RF_tmp.append(WW_RF)
                    i = i+1

                V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
                V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]

                W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
                W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]

                inputs2 = [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U]

                d_loss, _, _, _ = self.loss(inputs2)

            trainables = self.CNN_transmitter.trainable_weights + self.CNN_receiver.trainable_weights
            grads = tape.gradient(d_loss, trainables)
            self.optimizer.apply_gradients(zip(grads, trainables))
            loss_metric.update_state(d_loss)
            return {"neg_capacity_train_loss": loss_metric.result()}


    # see https://keras.io/api/models/model_training_apis/ for validation
    @tf.function
    def test_step(self, inputs0):
        if (self.phase_noise_exists_while_training == False):
            _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U, set_of_ns = inputs0
            # 5 4          6         5

            # rand_start = np.random.random_integers(low=0,
            #                                        high=self.Nsymb - self.number_of_OFDM_symbols_considered)
            # selected_symbols = range(rand_start, rand_start + self.number_of_OFDM_symbols_considered)
            selected_symbols = range(self.Nsymb)
            V_D_tmp = []
            V_RF_tmp = []
            W_D_tmp = []
            W_RF_tmp = []
            V_D, V_RF = self.CNN_transmitter(tf.tile(tf.expand_dims(H_tilde[:, 0, :, :, :, :], axis=1),
                                                     multiples=[1, round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1, 1]))  # batch, Nsymb, K, ... [only at ns=0]
            W_D, W_RF = self.CNN_receiver(tf.tile(tf.expand_dims(H_tilde[:, 0, :, :, :, :], axis=1),
                                                     multiples=[1, round(self.Nsymb * self.sampling_ratio_time_domain_keep), 1, 1, 1, 1]))
            i=0
            for ns in selected_symbols:
                VV_D, VV_RF = self.activation_TX([V_D[:,i,:], V_RF[:,i,:]])
                V_D_tmp.append(VV_D)
                V_RF_tmp.append(VV_RF)

                WW_D, WW_RF = self.activation_RX([W_D[:,i,:], W_RF[:,i,:]])  # batch, 1, K, ...
                W_D_tmp.append(WW_D)
                W_RF_tmp.append(WW_RF)
                i = i+1

            V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
            V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]
            W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
            W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]

            inputs2 = [V_D, W_D, tf.squeeze(H_tilde_complex[:,0,:,:,:]), V_RF, W_RF]
            d_loss = self.loss(inputs2)
            loss_metric_test.update_state(d_loss)

            # capacity metric
            V_D_tmp_ = []
            V_RF_tmp_ = []
            W_D_tmp_ = []
            W_RF_tmp_ = []
            # selected_symbols = range(self.Nsymb)
            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)
            V_D_, V_RF_ = self.CNN_transmitter(csi_tx)
            W_D_, W_RF_ = self.CNN_receiver(csi_rx)
            # for ns in range(self.Nsymb): # todo: return to this when done testing singleton ns set
            i = 0
            for ns in selected_symbols: # todo: return to this when done testing singleton ns set
                VV_D_, VV_RF_ = self.activation_TX([V_D_[:,i,:], V_RF_[:,i,:]])  # batch, 1, K, ...
                V_D_tmp_.append(VV_D_)
                V_RF_tmp_.append(VV_RF_)

                WW_D_, WW_RF_ = self.activation_RX([W_D_[:,i,:], W_RF_[:,i,:]])  # batch, 1, K, ...
                W_D_tmp_.append(WW_D_)
                W_RF_tmp_.append(WW_RF_)
                i = i+1

            V_D_ = tf.stack(V_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            V_RF_ = tf.stack(V_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]
            W_D_ = tf.stack(W_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            W_RF_ = tf.stack(W_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]
            capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise(
                [V_D_, W_D_, H_complex, V_RF_, W_RF_, Lambda_B, Lambda_U])
            capacity_metric_test.update_state(capacity_value)
            return {"neg_capacity_test_loss": loss_metric_test.result(),
                    'neg_capacity_performance_metric': capacity_metric_test.result()}
        else:
            _, H_complex, H_tilde, H_tilde_complex, Lambda_B, Lambda_U, set_of_ns = inputs0
            # 5 4          6         5
            selected_symbols = range(self.Nsymb)
            # rand_start = np.random.random_integers(low=0,
            #                                        high=self.Nsymb - self.number_of_OFDM_symbols_considered)
            # selected_symbols = range(rand_start, rand_start+ self.number_of_OFDM_symbols_considered)

            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)

            V_D_tmp = []
            V_RF_tmp = []
            W_D_tmp = []
            W_RF_tmp = []
            V_D, V_RF = self.CNN_transmitter(csi_tx)  # batch, Nsymb, K, ... [only at ns=0]
            W_D, W_RF = self.CNN_receiver(csi_rx)
            i = 0
            for ns in selected_symbols:
                VV_D, VV_RF = self.activation_TX([V_D[:,i,:], V_RF[:,i,:]])
                V_D_tmp.append(VV_D)
                V_RF_tmp.append(VV_RF)

                WW_D, WW_RF = self.activation_RX([W_D[:,i,:], W_RF[:,i,:]])  # batch, 1, K, ...
                W_D_tmp.append(WW_D)
                W_RF_tmp.append(WW_RF)
                i = i + 1

            V_D = tf.stack(V_D_tmp, axis=1)  # [should stack on axis ns]
            V_RF = tf.stack(V_RF_tmp, axis=1)  # [should stack on axis ns]
            W_D = tf.stack(W_D_tmp, axis=1)  # [should stack on axis ns]
            W_RF = tf.stack(W_RF_tmp, axis=1)  # [should stack on axis ns]

            inputs2 = [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U]
            d_loss, _, _, _ = self.loss(inputs2)
            loss_metric_test.update_state(d_loss)

            # capacity metric
            V_D_tmp_ = []
            V_RF_tmp_ = []
            W_D_tmp_ = []
            W_RF_tmp_ = []
            # selected_symbols = range(self.Nsymb)
            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)
            V_D_, V_RF_ = self.CNN_transmitter(csi_tx)
            W_D_, W_RF_ = self.CNN_receiver(csi_rx)

            i = 0
            for ns in selected_symbols:
                VV_D_, VV_RF_ = self.activation_TX([V_D_[:,i,:], V_RF_[:,i,:]])  # batch, 1, K, ...
                V_D_tmp_.append(VV_D_)
                V_RF_tmp_.append(VV_RF_)

                WW_D_, WW_RF_ = self.activation_RX([W_D_[:,i,:], W_RF_[:,i,:]])  # batch, 1, K, ...
                W_D_tmp_.append(WW_D_)
                W_RF_tmp_.append(WW_RF_)
                i = i + 1

            V_D_ = tf.stack(V_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            V_RF_ = tf.stack(V_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]
            W_D_ = tf.stack(W_D_tmp_, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            W_RF_ = tf.stack(W_RF_tmp_, axis=1)  # batch, Nsymb, ... [should stack on axis ns]

            capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise(
                [V_D_, W_D_, H_complex, V_RF_, W_RF_, Lambda_B, Lambda_U])
            capacity_metric_test.update_state(capacity_value)
            return {"neg_capacity_test_loss": loss_metric_test.result(),
                    'neg_capacity_performance_metric': capacity_metric_test.result()}

    @tf.function
    def evaluation_of_proposed_beamformer(self):
        obj_dataset_1 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                self.SNR, self.P, self.N_c, self.N_scatterers,
                                                self.angular_spread_rad, self.wavelength,
                                                self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                self.PHN_innovation_std, self.mat_fname, self.eval_dataset_size,
                                                self.eval_dataset_size, 'test',
                                                'yes')

        C_mean = 0
        # HH_complex = []
        # HH_tilde_0_cplx = []
        # LLambda_B = []
        # LLambda_U = []
        selected_symbols = range(self.Nsymb)
        # rand_start = np.random.random_integers(low=0, high=self.Nsymb - self.number_of_OFDM_symbols_considered)
        # selected_symbols = range(rand_start, rand_start + self.number_of_OFDM_symbols_considered)

        N_of_batches_in_DS = round(self.eval_dataset_size / self.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            H_complex, H_tilde, Lambda_B, Lambda_U, set_of_ns = \
                obj_dataset_1.data_generator_for_evaluation_of_proposed_beamformer(batch_number)

            csi_tx, csi_rx = self.NN_input_preparation(H_tilde)
            V_D, V_RF = self.CNN_transmitter(csi_tx)  # batch, Nsymb, K, ... [only at ns=0]
            W_D, W_RF = self.CNN_receiver(csi_rx)
            V_D_tmp = []
            V_RF_tmp = []
            W_D_tmp = []
            W_RF_tmp = []
            i = 0
            for ns in selected_symbols:
                VV_D, VV_RF = self.activation_TX([V_D[:,i,:], V_RF[:,i,:]])
                V_D_tmp.append(VV_D)
                V_RF_tmp.append(VV_RF)


                WW_D, WW_RF = self.activation_RX([W_D[:,i,:], W_RF[:,i,:]])  # batch, 1, K, ...
                W_D_tmp.append(WW_D)
                W_RF_tmp.append(WW_RF)
                i = i + 1

            V_D = tf.stack(V_D_tmp, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            V_RF = tf.stack(V_RF_tmp, axis=1)  # batch, Nsymb, ... [should stack on axis ns]

            W_D = tf.stack(W_D_tmp, axis=1)  # batch, Nsymb, K, ... [should stack on axis ns]
            W_RF = tf.stack(W_RF_tmp, axis=1)  # batch, Nsymb, ... [should stack on axis ns]
            T = self.metric_capacity_in_presence_of_phase_noise(
                [V_D, W_D, H_complex, V_RF, W_RF, Lambda_B, Lambda_U])
            C_mean = C_mean + T[0] / N_of_batches_in_DS

            if (batch_number == 0):
                c = T[1]
                RX = T[2]
                RQ = T[3]
            else:
                c = tf.concat([c, T[1]], axis=0)
                RX = tf.concat([RX, T[2]], axis=0)
                RQ = tf.concat([RQ, T[3]], axis=0)

        #     # Creating data for Sohrabis method in Matlab
        #
        #     HH_complex.append(H_complex)
        #     HH_tilde_0_cplx.append(tf.complex(H_tilde_0[:, :, :, :, 0], H_tilde_0[:, :, :, :, 1]))
        #     LLambda_B.append(Lambda_B)
        #     LLambda_U.append(Lambda_U)
        #
        # HHH_complex = tf.concat(HH_complex, axis=0).numpy()
        # HHH_tilde_0 = tf.concat(HH_tilde_0_cplx, axis=0).numpy()
        # LLLambda_B = tf.concat(LLambda_B, axis=0).numpy()
        # LLLambda_U = tf.concat(LLambda_U, axis=0).numpy()
        #
        # mdic = {"H": HHH_complex,
        #         "H_tilde_0": HHH_tilde_0,
        #         'Lambda_B': LLLambda_B,
        #         'Lambda_U': LLLambda_U}
        # sio.savemat("data_set_for_matlab.mat", mdic)

        return C_mean, c, RX, RQ

    @tf.function
    def evaluation_of_Sohrabis_beamformer(self):
        dataset_for_testing_sohrabi = 'DS_for_py_for_testing_Sohrabi.mat'

        obj_dataset_2 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                self.SNR, self.P, self.N_c, self.N_scatterers,
                                                self.angular_spread_rad, self.wavelength,
                                                self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                self.PHN_innovation_std, self.mat_fname, self.eval_dataset_size,
                                                self.eval_dataset_size, 'test',
                                                'yes')
        C_mean = 0
        N_of_batches_in_DS = round(self.eval_dataset_size / self.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            # print('batch_number: ', batch_number)
            H_complex, Lambda_B, Lambda_U, V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, \
            V_D_Sohrabi_optimized, W_D_Sohrabi_optimized = \
                obj_dataset_2.data_generator_for_evaluation_of_Sohrabis_beamformer(batch_number)
            # print('in ml model:', V_D_Sohrabi_optimized.shape, W_D_Sohrabi_optimized.shape,H_complex.shape,V_RF_Sohrabi_optimized.shape,W_RF_Sohrabi_optimized.shape,Lambda_B.shape,Lambda_U.shape)
            T = self.metric_capacity_in_presence_of_phase_noise([V_D_Sohrabi_optimized,
                                                                 W_D_Sohrabi_optimized,
                                                                 H_complex,
                                                                 V_RF_Sohrabi_optimized,
                                                                 W_RF_Sohrabi_optimized,
                                                                 Lambda_B,
                                                                 Lambda_U])
            C_mean = C_mean + T[0] / N_of_batches_in_DS

            if (batch_number == 0):
                c = T[1]
                RX = T[2]
                RQ = T[3]
            else:
                c = tf.concat([c, T[1]], axis=0)
                RX = tf.concat([RX, T[2]], axis=0)
                RQ = tf.concat([RQ, T[3]], axis=0)

        return C_mean, c, RX, RQ

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_metric, loss_metric_test, capacity_metric_test]
