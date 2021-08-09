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

    def __init__(self, model_dnn, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, phase_shift_stddiv, truncation_ratio_keep, Nsymb, Ts, fc, c, PHN_innovation_std, mat_fname, eval_dataset_size, mode):
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

    def compile(self, optimizer, loss, activation, phase_noise, metric_capacity_in_presence_of_phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.metric_capacity_in_presence_of_phase_noise = metric_capacity_in_presence_of_phase_noise
        self.activation = activation
        self.phase_noise = phase_noise


    def train_step(self, inputs0):
        H_complex, H = inputs0
        with tf.GradientTape() as tape:
            V_D, W_D, V_RF, W_RF = self.model_dnn(H)
            inputs1 = [V_D, W_D, V_RF, W_RF]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            inputs2 = [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx]
            d_loss = self.loss(inputs2)
        grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))
        loss_metric.update_state(d_loss)
        return {"neg_capacity_train_loss": loss_metric.result()}

    # see https://keras.io/api/models/model_training_apis/ for validation

    def test_step(self, inputs0):
        H_complex, H_tilde_0, Lambda_B, Lambda_U = inputs0
        V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(H_tilde_0, training=False)
        inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
        V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
        loss_val= self.loss([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx])
        # print(loss_val)
        loss_metric_test.update_state(loss_val)
        capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
        capacity_metric_test.update_state(capacity_value)
        # print(capacity_value)
        return {"neg_capacity_test_loss": loss_metric_test.result() , 'neg_capacity_performance_metric': capacity_metric_test.result()}


    def evaluation_of_proposed_beamformer(self):
        obj_dataset_1 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                   self.SNR, self.P, self.N_c, self.N_scatterers,
                                                   self.angular_spread_rad, self.wavelength,
                                                   self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                   self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                   self.PHN_innovation_std, self.mat_fname, self.eval_dataset_size, 'test')
        C = 0
        C_tmp = []
        RX_tmp = []
        RQ_tmp = []
        HH_complex = []
        HH_tilde_0_cplx = []
        LLambda_B = []
        LLambda_U = []
        N_of_batches_in_DS = int(self.eval_dataset_size/self.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            print('batch_number: ', batch_number)
            H_complex, H_tilde_0, Lambda_B, Lambda_U = obj_dataset_1.data_generator_for_evaluation_of_proposed_beamformer(batch_number)

            V_D, W_D, V_RF, W_RF = self.model_dnn(H_tilde_0, training=False)
            inputs1 = [V_D, W_D, V_RF, W_RF]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            T = self.metric_capacity_in_presence_of_phase_noise([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            C = C + T[0]/N_of_batches_in_DS
            C_tmp.append(T[1])
            RX_tmp.append(T[2])
            RQ_tmp.append(T[3])


        # # Creating data for Sohrabis method in Matlab

            # HH_complex.append(H_complex)
            # HH_tilde_0_cplx.append(tf.complex(H_tilde_0[:, :, :, :, 0], H_tilde_0[:, :, :, :, 1]))
            # LLambda_B.append(Lambda_B)
            # LLambda_U.append(Lambda_U)

        # HHH_complex = tf.concat(HH_complex, axis=0).numpy()
        # HHH_tilde_0 = tf.concat(HH_tilde_0_cplx, axis=0).numpy()
        # LLLambda_B = tf.concat(LLambda_B, axis=0).numpy()
        # LLLambda_U = tf.concat(LLambda_U, axis=0).numpy()
        #
        # mdic = {"H": HHH_complex,
        #         "H_tilde_0": HHH_tilde_0,
        #         'Lambda_B': LLLambda_B,
        #         'Lambda_U': LLLambda_U}
        #
        # sio.savemat("C:/Users/jabba/Google Drive/Main/Codes/ML_MIMO_new_project/PY_projects/convnet_transfer_learning_v1/data_set_for_matlab.mat",mdic)

        return tf.squeeze(C),\
               tf.squeeze(tf.concat(C_tmp, axis = 0)),\
               tf.squeeze(tf.concat(tf.math.real(RX_tmp), axis = 0)),\
               tf.squeeze(tf.concat( tf.math.real(RQ_tmp), axis = 0))



    def evaluation_of_Sohrabis_beamformer(self):
        dataset_for_testing_sohrabi = 'C:/Users/jabba/Videos/datasets/DS_for_py_for_testing_Sohrabi.mat'
        obj_dataset_2 = dataset_generator_class(self.N_b_a, self.N_b_rf, self.N_u_a, self.N_u_rf, self.N_s, self.K,
                                                   self.SNR, self.P, self.N_c, self.N_scatterers,
                                                   self.angular_spread_rad, self.wavelength,
                                                   self.d, self.BATCHSIZE, self.phase_shift_stddiv,
                                                   self.truncation_ratio_keep, self.Nsymb, self.Ts, self.fc, self.c,
                                                   self.PHN_innovation_std, dataset_for_testing_sohrabi, self.eval_dataset_size,
                                                   'test')
        C = 0
        C_tmp = []
        RX_tmp = []
        RQ_tmp = []
        N_of_batches_in_DS = int(self.eval_dataset_size/self.BATCHSIZE)
        for batch_number in range(N_of_batches_in_DS):
            print('batch_number: ', batch_number)
            H_complex, Lambda_B, Lambda_U, \
            V_RF_Sohrabi_optimized, W_RF_Sohrabi_optimized, \
            V_D_Sohrabi_optimized, W_D_Sohrabi_optimized = obj_dataset_2.data_generator_for_evaluation_of_Sohrabis_beamformer(batch_number)
            # print('V_D_Sohrabi_optimized:   ', V_D_Sohrabi_optimized.shape)
            # print('norm of V_D_new: ', tf.norm(tf.squeeze(V_D_new[0,:])))
            # print('norm of W_D_cplx: ', tf.norm(tf.squeeze(W_D_cplx[0,:])))
            # print('norm of V_RF_cplx: ', tf.norm(tf.squeeze(V_RF_cplx[0,:])))
            # print('norm of W_RF_cplx: ', tf.norm(tf.squeeze(W_RF_cplx[0,:])))
            #
            # print('V_D_new: ', (tf.squeeze(V_D_new[0,:])))
            # print('W_D_cplx: ', (tf.squeeze(W_D_cplx[0,:])))
            # print('V_RF_cplx: ', (tf.squeeze(V_RF_cplx[0,:])))
            # print('W_RF_cplx: ', (tf.squeeze(W_RF_cplx[0,:])))

            T = self.metric_capacity_in_presence_of_phase_noise([V_D_Sohrabi_optimized, W_D_Sohrabi_optimized,
                                                                 H_complex,V_RF_Sohrabi_optimized,
                                                                 W_RF_Sohrabi_optimized , Lambda_B, Lambda_U])

            C = C + T[0] / N_of_batches_in_DS
            C_tmp.append(T[1])
            RX_tmp.append(T[2])
            RQ_tmp.append(T[3])

        return tf.squeeze(C),\
               tf.squeeze(tf.concat(C_tmp, axis = 0)),\
               tf.squeeze(tf.concat(tf.math.real(RX_tmp), axis = 0)),\
               tf.squeeze(tf.concat( tf.math.real(RQ_tmp), axis = 0))
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_metric, loss_metric_test, capacity_metric_test]
