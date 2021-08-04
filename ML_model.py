import tensorflow as tf


loss_metric = tf.keras.metrics.Mean(name='neg_capacity')
loss_metric_test = tf.keras.metrics.Mean(name='neg_capacity_test')
norm_records = tf.keras.metrics.Mean(name='norm')
capacity_metric_test = tf.keras.metrics.Mean(name='neg_capacity_performance_metric')


class ML_model_class(tf.keras.Model):

    def __init__(self, model_dnn):
        super(ML_model_class, self).__init__()
        self.model_dnn = model_dnn

    def compile(self, optimizer, loss, activation, phase_noise, metric_capacity_in_presence_of_phase_noise):
        super(ML_model_class, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.metric_capacity_in_presence_of_phase_noise = metric_capacity_in_presence_of_phase_noise
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
            H_complex, H = inputs0
            with tf.GradientTape() as tape:
                # Unpack the data
                V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(H)
                inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
                V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
                inputs2 = [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx]
                # inputs2 = inputs1
                d_loss = self.loss(inputs2)
                # print(d_loss.shape)
            grads = tape.gradient(d_loss, self.model_dnn.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model_dnn.trainable_weights))
            loss_metric.update_state(d_loss)

            return {"neg_capacity_train_loss": loss_metric.result()}

    # see https://keras.io/api/models/model_training_apis/ for validation
    @tf.function
    def test_step(self, inputs0):
        if (self.phase_noise == 'y'):
            H_complex, csi_tilde_0, PHN_B, PHN_U = inputs0
            # Unpack the data
            # print(csi_reduced.shape)
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(csi_tilde_0, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            inputs1 = [V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, PHN_B, PHN_U]
            test_loss, _, _, _ = self.loss(inputs1)
            # print(test_loss.shape)
            loss_metric_test.update_state(test_loss)

            return {"neg_capacity_test_loss": loss_metric_test.result()}
        else:
            H_complex, H_tilde_0, Lambda_B, Lambda_U = inputs0
            V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx = self.model_dnn(H_tilde_0, training=False)
            inputs1 = [V_D_cplx, W_D_cplx, V_RF_cplx, W_RF_cplx]
            V_D_new, W_D_cplx, V_RF_cplx, W_RF_cplx = self.activation(inputs1)
            loss_metric_test.update_state(self.loss([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx]))
            capacity_value, _, _, _ = self.metric_capacity_in_presence_of_phase_noise([V_D_new, W_D_cplx, H_complex, V_RF_cplx, W_RF_cplx, Lambda_B, Lambda_U])
            capacity_metric_test.update_state(capacity_value)

            return {"neg_capacity_test_loss": loss_metric_test.result() , 'neg_capacity_performance_metric': capacity_metric_test.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_metric, loss_metric_test, capacity_metric_test]