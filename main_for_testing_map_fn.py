# Imports libs /////////////////////////////////////////////////////////////////////////////////////////////////////////
import datetime
import time
import scipy.io as sio
import tensorflow as tf
import numpy as np






#
# def custom_tf_while_loop():
#
#     # Track both the loop index and summation in a tuple in the form (index, summation)
#     loop_index = tf.constant(0)
#     loop_summation = tf.constant(0.0)
#
#     def loop_condition(loop_index, loop_summation):
#         loop_threshold = 5
#         return tf.less(loop_index, loop_threshold)
#
#     # The loop body, this will return a result tuple in the same form (index, summation)
#     def loop_body(loop_index, loop_summation):
#        loop_summation = loop_summation + tf.cast(loop_index, tf.float32)
#
#        loop_index = tf.add(loop_index, 1)
#        return loop_index, loop_summation
#
#     # We do not care about the index value here, return only the summation
#     return tf.while_loop(loop_condition, loop_body, (loop_index, loop_summation))[1]
#



def custom_tf_while_loop():

    # Track both the loop index and summation in a tuple in the form (index, summation)
    loop_index = tf.constant(0)
    loop_output = tf.constant(0.0)

    def loop_condition(loop_index, loop_summation):
        loop_threshold = 100
        return tf.less(loop_index, loop_threshold)

    # The loop body, this will return a result tuple in the same form (index, summation)
    def loop_body(loop_index, loop_output):
        x = tf.random.uniform([1000, 1000])
        loop_output = tf.matmul(x, x)

        loop_index = tf.add(loop_index, 1)
        return loop_index, loop_output

    # We do not care about the index value here, return only the summation
    return tf.while_loop(loop_condition, loop_body, (loop_index, loop_output))[1]

def test_func(x):
    return tf.matmul(x,x)
    

# Main /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    n = 100
    x = tf.random.uniform([1000, 1000])
    start = time.time()
    for i in range(n):
        tf.matmul(x, x)
    end = time.time()
    print("________------------_______________---------------____________------------for loop implementation elapsed time:", 1000 * (end - start), "ms")


    x2 = tf.random.uniform([n, 1000, 1000])
    start = time.time()
    tf.map_fn(test_func, elems= x2)
    end = time.time()
    print("________------------_______________---------------____________------------map_fn implementation elapsed time:", 1000 * (end - start), "ms")

    start = time.time()
    tf.vectorized_map(test_func, elems= x2)
    end = time.time()
    print("________------------_______________---------------____________------------VECTORIZED_MAP implementation elapsed time:", 1000 * (end - start), "ms")

    start = time.time()
    r = custom_tf_while_loop()
    # print(r)
    end = time.time()
    print("________------------_______________---------------____________------------tf.while implementation elapsed time:", 1000 * (end - start), "ms")
