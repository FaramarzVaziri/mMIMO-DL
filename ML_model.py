import tensorflow as tf
import numpy as np
import scipy.io as sio
import tensorflow as tf
from dataset_generator import dataset_generator_class

loss_metric = tf.keras.metrics.Mean(name='neg_capacity')
loss_metric_test = tf.keras.metrics.Mean(name='neg_capacity_test')
norm_records = tf.keras.metrics.Mean(name='norm')
capacity_metric_test = tf.keras.metrics.Mean(name='neg_capacity_performance_metric')

class ML_model_class(tf.keras.Model):

    def __init__(self, model_dnn, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad,
                 wavelength, d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c,
                 PHN_innovation_std, mat_fname, eval_dataset_size, mode, on_what_device,
                 phase_noise_exists_while_training):
        super(ML_model_class, self).__init__()
        self.model_dnn = model_dnn
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
        self.eval_dataset_size = eval_dataset_size
        self.PHN_innovation_std = PHN_innovation_std
        self.mode = mode
        self.on_what_device = on_what_device
        self.phase_noise_exists_while_training = phase_noise_exists_while_training

    def compile(self, optimizer, loss, activation, phase_noise, metric_capacity_in_presence_of_phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.metric_capacity_in_presence_of_phase_noise = metric_capacity_in_presence_of_phase_noise
        self.activation = activation
        self.phase_noise = phase_noise

    
    def train_step(self, inputs0):
        if (self.phase_noise_exists_while_training == False):
            H_tilde_0_complex, H_tilde_0, H_complex, Lambda_B, Lambda_U = inputs0
            with tf.GradientTape() as tape:
                V_D, W_D, V_RF, W_RF = self.model_dnn(H_tilde_0)
                inputs1 = [V_D, W_D, V_RF, W_RF]
                V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
                inputs2 = [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx]
                d_loss = self.loss(inputs2)
            grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))
            loss_metric.update_state(d_loss)
            return {"neg_capacity_train_loss": loss_metric.result()}
        else:
            _, H_tilde_0, H_complex, Lambda_B, Lambda_U = inputs0
            with tf.GradientTape() as tape:
                V_D, W_D, V_RF, W_RF = self.model_dnn(H_tilde_0)
                inputs1 = [V_D, W_D, V_RF, W_RF]
                V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
                inputs2 = [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U]
                d_loss = self.loss(inputs2)
            grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))
            loss_metric.update_state(d_loss)
            return {"neg_capacity_train_loss": loss_metric.result()}

    # see https://keras.io/api/models/model_training_apis/ for validation
    
    def test_step(self, inputs0):
        if (self.phase_noise_exists_while_training == False):
            H_tilde_0_complex, H_tilde_0, H_complex, Lambda_B, Lambda_U = inputs0
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(H_tilde_0, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            loss_val= self.loss([V_D_new, W_D_cplx, H_tilde_0_complex, V_RF_cplx, W_RF_cplx])
            loss_metric_test.update_state(loss_val)
            capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            capacity_metric_test.update_state(capacity_value)
            return {"neg_capacity_test_loss": loss_metric_test.result() , 'neg_capacity_performance_metric': capacity_metric_test.result()}
        else:
            _, H_tilde_0, H_complex, Lambda_B, Lambda_U = inputs0
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(H_tilde_0, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            loss_val = self.loss([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            loss_metric_test.update_state(loss_val)
            capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise(
                [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            capacity_metric_test.update_state(capacity_value)
            return {"neg_capacity_test_loss": loss_metric_test.result(),
                    'neg_capacity_performance_metric': capacity_metric_test.result()}

    def evaluation_of_proposed_beamformer(self):
        obj_dataset_1 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                   self.SNR, self.P, self.N_c, self.N_scatterers,
                                                   self.angular_spread_rad, self.wavelength,
                                                   self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                   self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                   self.PHN_innovation_std, self.mat_fname, self.eval_dataset_size, 'test', 'yes')
        C_mean = 0

        HH_complex = []
        HH_tilde_0_cplx = []
        LLambda_B = []
        LLambda_U = []
        N_of_batches_in_DS = int(self.eval_dataset_size/self.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            # print('batch_number: ', batch_number)
            H_complex, H_tilde_0, Lambda_B, Lambda_U = obj_dataset_1.data_generator_for_evaluation_of_proposed_beamformer(batch_number)

            V_D, W_D, V_RF, W_RF = self.model_dnn(H_tilde_0, training=False)
            inputs1 = [V_D, W_D, V_RF, W_RF]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            T = self.metric_capacity_in_presence_of_phase_noise([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            C_mean = C_mean + T[0]/N_of_batches_in_DS

            if (batch_number == 0):
                c = T[1]
                RX = T[2]
                RQ = T[3]
            else:
                c = tf.concat([c, T[1]], axis=0)
                RX = tf.concat([RX, T[2]], axis=0)
                RQ = tf.concat([RQ, T[3]], axis=0)

        # Creating data for Sohrabis method in Matlab

            HH_complex.append(H_complex)
            HH_tilde_0_cplx.append(tf.complex(H_tilde_0[:, :, :, :, 0], H_tilde_0[:, :, :, :, 1]))
            LLambda_B.append(Lambda_B)
            LLambda_U.append(Lambda_U)

        HHH_complex = tf.concat(HH_complex, axis=0).numpy()
        HHH_tilde_0 = tf.concat(HH_tilde_0_cplx, axis=0).numpy()
        LLLambda_B = tf.concat(LLambda_B, axis=0).numpy()
        LLLambda_U = tf.concat(LLambda_U, axis=0).numpy()

        mdic = {"H": HHH_complex,
                "H_tilde_0": HHH_tilde_0,
                'Lambda_B': LLLambda_B,
                'Lambda_U': LLLambda_U}
        if (self.on_what_device == 'cpu'):
            sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/PY_projects/convnet_transfer_learning_v1/data_set_for_matlab.mat",mdic)
        else:
            sio.savemat("/data/jabbarva/github_repo/mMIMO-DL/datasets/data_set_for_matlab.mat",mdic)

        return C_mean, c, RX, RQ

    
    def evaluation_of_Sohrabis_beamformer(self):
        if (self.on_what_device == 'cpu'):
            dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'
        else:
            dataset_for_testing_sohrabi = '/data/jabbarva/github_repo/mMIMO-DL/datasets/DS_for_py_for_testing_Sohrabi.mat'

        obj_dataset_2 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                   self.SNR, self.P, self.N_c, self.N_scatterers,
                                                   self.angular_spread_rad, self.wavelength,
                                                   self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                   self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                   self.PHN_innovation_std, dataset_for_testing_sohrabi, self.eval_dataset_size,
                                                   'test', 'yes')
        C_mean = 0
        N_of_batches_in_DS = int(self.eval_dataset_size / self.BATCHSIZE)
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
