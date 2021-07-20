import tensorflow as tf


loss_metric = tf.keras.metrics.Mean(name='neg_capacity')
loss_metric_test = tf.keras.metrics.Mean(name='neg_capacity_test')
norm_records = tf.keras.metrics.Mean(name='norm')


class ML_model_class(tf.keras.Model):

    def __init__(self, model_dnn):
        super(ML_model_class, self).__init__()
        self.model_dnn = model_dnn

    def compile(self, optimizer, loss, activation, phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.activation = activation
        self.phase_noise = phase_noise

    @tf.function
    def train_step(self, inputs0):
        if (self.phase_noise == 'y'):
            # H_complex_dataset, H_tilde_0_dataset, Lambda_B_dataset, Lambda_U_dataset
            csi_original, csi_tilde_0, PHN_B, PHN_U = inputs0
            with tf.GradientTape() as tape:
                # Unpack the data
                V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(csi_tilde_0)
                inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
                V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
                # print(V_D_new.shape)
                inputs1 = [V_D_new, W_D_cplx, csi_original, V_RF_cplx, W_RF_cplx, PHN_B, PHN_U]
                d_loss, _, _, _ = self.loss(inputs1)
                # print(d_loss.shape)
            grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))

            loss_metric.update_state(d_loss)

            return {"neg_capacity": loss_metric.result()}
        else:
            # H_complex_dataset, H_tilde_0_dataset, Lambda_B_dataset, Lambda_U_dataset
            csi_original = inputs0
            with tf.GradientTape() as tape:
                # Unpack the data
                V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(csi_original)
                inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
                V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
                # print(V_D_new.shape)
                inputs1 = [V_D_new, W_D_cplx, csi_original, V_RF_cplx, W_RF_cplx]
                d_loss = self.loss(inputs1)
                # print(d_loss.shape)
            grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)

            self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))

            loss_metric.update_state(d_loss)

            return {"neg_capacity": loss_metric.result()}

    # see https://keras.io/api/models/model_training_apis/ for validation
    @tf.function
    def test_step(self, inputs0):
        if (self.phase_noise == 'y'):
            csi_original, csi_tilde_0, PHN_B, PHN_U = inputs0
            # Unpack the data
            # print(csi_reduced.shape)
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(csi_tilde_0, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            # print(V_D_new.shape)
            # TT0 = tf.linalg.matmul(V_RF_cplx[0,:,:] , V_D_new[0,0,:,:] , adjoint_a=False , adjoint_b=False)
            # TT1 = tf.linalg.matmul(TT0 , TT0, adjoint_a=False, adjoint_b=True)
            # norm = tf.linalg.trace(TT1)
            # norm_records.update_state(norm)

            inputs1 = [V_D_new, W_D_cplx, csi_original, V_RF_cplx, W_RF_cplx, PHN_B, PHN_U]
            test_loss, _, _, _ = self.loss(inputs1)
            # print(test_loss.shape)
            loss_metric_test.update_state(test_loss)

            return {"neg_capacity_test": loss_metric_test.result()}  # , "norm": norm_records.result()}
        else:
            csi_original = inputs0
            # Unpack the data
            # print(csi_reduced.shape)
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(csi_original, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            # print(V_D_new.shape)
            # TT0 = tf.linalg.matmul(V_RF_cplx[0,:,:] , V_D_new[0,0,:,:] , adjoint_a=False , adjoint_b=False)
            # TT1 = tf.linalg.matmul(TT0 , TT0, adjoint_a=False, adjoint_b=True)
            # norm = tf.linalg.trace(TT1)
            # norm_records.update_state(norm)

            inputs1 = [V_D_new, W_D_cplx, csi_original, V_RF_cplx, W_RF_cplx]
            test_loss = self.loss(inputs1)
            # print(test_loss.shape)
            loss_metric_test.update_state(test_loss)

            return {"neg_capacity_test": loss_metric_test.result()}  # , "norm": norm_records.result()}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_metric, loss_metric_test]    