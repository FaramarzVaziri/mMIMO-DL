# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np

# tf.config.run_functions_eagerly(True)
# import matplotlib.pyplot as plt
# tf.distribute.Strategy

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> the device name: ',
      tf.config.list_physical_devices('GPU'))
if tf.test.gpu_device_name() == '/device:GPU:0':
    tf.device('/device:GPU:0')

# Import classes ///////////////////////////////////////////////////////////////////////////////////////////////////////
from CNN_model import CNN_model_class
from ML_model import ML_model_class
from Sohrabi_s_method_tester import Sohrabi_s_method_tester_class
from dataset_generator import dataset_generator_class
from loss_parallel_phase_noise_free import loss_parallel_phase_noise_free_class
from loss_parallel_phase_noised import paralle_loss_phase_noised_class


def test_func(x):
    return tf.matmul(x, x)


# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':

    print('for loop implementation')
    n=1000
    x = tf.random.uniform([1000, 1000])
    start = time.time()
    for i in range(n):
        tf.matmul(x, x)
    end = time.time()
    print("elapsed time:", 1000 * (end - start), "ms")


    print('map_fn implementation')
    x2 = tf.random.uniform([1000, 1000, 1000])
    start = time.time()
    tf.map_fn(test_func, elems= x2)
    end = time.time()
    print("elapsed time:", 1000 * (end - start), "ms")

    print('VECTORIZED_MAP implementation')
    x2 = tf.random.uniform([1000, 1000, 1000])
    start = time.time()
    tf.vectorized_map(test_func, elems= x2)
    end = time.time()
    print("elapsed time:", 1000 * (end - start), "ms")
