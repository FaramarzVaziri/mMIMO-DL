import numpy as np
import scipy.io as sio
import tensorflow as tf
from os.path import dirname, join as pjoin


#
# if tf.test.gpu_device_name() == '/device:GPU:0':
#   tf.device('/device:GPU:0')

class TFrecord_generation_class:
    def __init__(self, N_b_a, N_b_rf, N_u_a, N_u_rf, N_s, K, SNR, P, N_c, N_scatterers, angular_spread_rad, wavelength,
                 d, BATCHSIZE, truncation_ratio_keep, Nsymb, Ts, fc, c, PHN_innovation_std):
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
        self.truncation_ratio_keep = truncation_ratio_keep
        self.Nsymb = Nsymb
        self.Ts = Ts
        self.fc = fc
        self.c = c
        self.PHN_innovation_std = PHN_innovation_std

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, H):
        # Creates a tf.train.Example message ready to be written to a file.
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {'H': self._float_feature(H)}
        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.features(feature=feature))
        return example_proto.SerializeToString()
