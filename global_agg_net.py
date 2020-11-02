import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import config_res as config
from cnn_utils_res import *

import resnet_rgb_model as model
import resnet_depth_model as model_depth

batch_size = config.net_params['batch_size']
current_epoch = config.net_params['load_epoch']
DISP_SCALING = 10
MIN_DISP = 0.01
LSTM_NUM = 2
LSTM_HIDDEN_SIZE = 256
time_seq = 3

def End_Net_weights_init():

    """
    Initialize Aggregation Network Weights and Summaries
    """

    W_ext1_rot = weight_variable([3, 3, 768, 384], "_8")
    W_ext2_rot = weight_variable([3, 3, 384, 384], "_9")
    W_ext3_rot = weight_variable([1, 2, 384, 384], "_10")

    W_ext4_rot = weight_variable([1, 1, 384, 384], "_11")
    W_fc_rot = weight_variable_fc([3840, 3], "_12")

    W_ext1_tr = weight_variable([3, 3, 768, 384], "_13")
    W_ext2_tr = weight_variable([3, 3, 384, 384], "_14")
    W_ext3_tr = weight_variable([1, 2, 384, 384], "_15")
    W_ext4_tr = weight_variable([1, 1, 384, 384], "_16")
    W_fc_tr = weight_variable_fc([3840, 3], "_17")
    end_weights = [W_ext1_rot, W_ext2_rot, W_ext3_rot, W_ext4_rot, W_fc_rot, W_ext1_tr, W_ext2_tr, W_ext3_tr, W_ext4_tr, W_fc_tr]

    weight_summaries = []

    for weight_index in range(len(end_weights)):
        with tf.name_scope('weight_%d'%weight_index):
            weight_summaries += variable_summaries(end_weights[weight_index])

    return end_weights, weight_summaries

def End_Net(input_x, phase_depth, keep_prob):

    """
    Define Aggregation Network
    """

    weights, summaries = End_Net_weights_init()

    layer8_rot = conv2d_batchnorm_init(input_x, weights[0], name="conv_9", phase= phase_depth, stride=[1,2,2,1])
    layer9_rot = conv2d_batchnorm_init(layer8_rot, weights[1], name="conv_10", phase= phase_depth, stride=[1,2,2,1])
    layer10_rot = conv2d_batchnorm_init(layer9_rot, weights[2], name="conv_11", phase= phase_depth, stride=[1,1,1,1])

    layer11_rot = conv2d_batchnorm_init(layer10_rot, weights[3], name="conv_12", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_rot = tf.reshape(layer11_rot, [batch_size, 3840])
    # layer11_m_rot = tf.reshape(layer11_rot, [batch_size, 2560])
    layer11_drop_rot = tf.nn.dropout(layer11_m_rot, keep_prob)
    layer11_vec_rot = (tf.matmul(layer11_drop_rot, weights[4]))

    layer8_tr = conv2d_batchnorm_init(input_x, weights[5], name="conv_13", phase=phase_depth, stride=[1, 2, 2, 1])
    layer9_tr = conv2d_batchnorm_init(layer8_tr, weights[6], name="conv_14", phase=phase_depth, stride=[1, 2, 2, 1])
    layer10_tr = conv2d_batchnorm_init(layer9_tr, weights[7], name="conv_15", phase=phase_depth, stride=[1, 1, 1, 1])

    layer11_tr = conv2d_batchnorm_init(layer10_tr, weights[8], name="conv_16", phase= phase_depth, stride=[1,1,1,1])
    layer11_m_tr = tf.reshape(layer11_tr, [batch_size, 3840])
    # layer11_m_tr = tf.reshape(layer11_tr, [batch_size, 2560])
    layer11_drop_tr = tf.nn.dropout(layer11_m_tr, keep_prob)
    layer11_vec_tr = (tf.matmul(layer11_drop_tr, weights[9]))

    output_vectors = tf.concat([layer11_vec_tr, layer11_vec_rot], 1)
    return output_vectors




def End_Net_Out(cam1, cam2, cam3, velo1, velo2, velo3, phase_rgb, phase, keep_prob):

    """
    Computation Graph
    """
    we = np.array([0.1, 0.2, 0.7], np.float32)
    cam = tf.concat([cam1,cam2,cam3], 0)
    velo = tf.concat([velo1,velo2,velo3], 0)
    RGB_Net_obj = model.Resnet(cam, phase_rgb)
    Depth_Net_obj = model_depth.Depthnet(velo, phase)

    with tf.variable_scope('ResNet'):
        output_rgb = RGB_Net_obj.Net()

    with tf.variable_scope('DepthNet'):
        output_depth = Depth_Net_obj.Net()

    with tf.variable_scope('Concat'):
        layer_next1 = tf.concat([output_rgb[:batch_size], output_depth[:batch_size]], 3)

        layer_next2 = tf.concat([output_rgb[batch_size:2 * batch_size], output_depth[batch_size:2 * batch_size]], 3)

        layer_next3 = tf.concat([output_rgb[2 * batch_size:], output_depth[2 * batch_size:]], 3)

        feature_concat = tf.concat([layer_next1, layer_next2, layer_next3], 0)

    W_ext1 = weight_variable([3, 3, 768, 384], "_8")

    layer8 = conv2d_batchnorm_init(feature_concat, W_ext1, name="conv_9", phase=phase, stride=[1, 2, 2, 1]) # (?, 3, 10, 384)

    with tf.variable_scope('LSTM'):
        lstm_inputs = tf.reshape(layer8, [3, batch_size, 3, 10, 384])

        lstm_inputs = tf.unstack(lstm_inputs, axis=0)
        lstm_inputs = [tf.contrib.layers.flatten(lstm_inputs[i], [-1, ]) for i in range(len(lstm_inputs))]
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(LSTM_HIDDEN_SIZE) for _ in range(LSTM_NUM)]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        lstm_outputs, lstm_state = tf.nn.static_rnn(cell=multi_rnn_cell,
                                                    inputs=lstm_inputs,
                                                    dtype=tf.float32)

    with tf.variable_scope('full_connect'):
        fc0 = [tf.nn.dropout(fs, keep_prob) for fs in lstm_outputs]
        fc_layer1 = [tf.layers.dense(o, 128, activation=tf.nn.relu) for o in fc0]
        xs = [tf.nn.dropout(fs, keep_prob) for fs in fc_layer1]
        trs = [tf.layers.dense(x, 3) for x in xs]
        rots = [tf.layers.dense(x, 3) for x in xs]
        trs = [trs[i] * we[i] for i in range(3)]
        rots = [rots[i] * we[i] for i in range(3)]
        tr = tf.reduce_sum(trs, 0)
        rot = tf.reduce_sum(rots, 0)
        output_vectors = tf.concat([tr, rot], 1)

    return output_vectors
