import numpy as np
import tensorflow as tf
import config_res as config
import argparse
import os
import cv2
from utils import *
from loss_fun import *
import data_load as dl
# import model_utils
import global_agg_net
import transform_function
import all_transformer as at


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default='0')
    args = parser.parse_args()
    return args


def sess_config(args=None):
    log_device_placement = False  # 是否打印设备分配日志
    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85, allow_growth= True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用 GPU 0
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            gpu_options=gpu_options)

    return config


IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']

batch_size = config.net_params['batch_size']
learning_rate = config.net_params['learning_rate']
n_epochs = config.net_params['epochs']
current_epoch = config.net_params['load_epoch']


tf.reset_default_graph()
cam1 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 3), name = "cam1")
cam2 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 3), name = "cam2")
cam3 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 3), name = "cam3")

velo1 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 1), name = "velo1")
velo2 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 1), name = "velo2")
velo3 = tf.placeholder(tf.float32, shape = (None, IMG_HT, IMG_WDT, 1), name = "velo3")

velo_pose = tf.placeholder(tf.float32,shape=(None, 4, 4),name = 'velo_pose')
expected_vector = tf.placeholder(tf.float32,shape=(None,6),name = 'expected_calib')
match_points = tf.placeholder(tf.float32,shape=(None, 100, 4),name = 'match_points')
decalib_mat = tf.placeholder(tf.float32,shape=(None, 4, 4),name = 'decalib_mat')
p_rect0 = tf.placeholder(tf.float32,shape=(None, 4, 4),name = 'p_rect0')
velo_to_cams = tf.placeholder(tf.float32,shape=(None, 4, 4),name = 'velo_to_cam')
intrinsics = tf.placeholder(tf.float32,shape=(None, 3, 3),name = 'intrinsics')

x = tf.placeholder(tf.int32, shape=(None), name = 'x')

keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

checkpoint_path = config.paths['checkpoint_path']
total_train = config.net_params['total_train']
load_batch = config.net_params['load_batch']
training_imgs_path = config.paths['training_imgs']
validation_imgs_path = config.paths['validation_imgs']
total_partitions_train = int(total_train / (load_batch * batch_size))


sift_match_mask = tf.map_fn(lambda x: get_sift_match_mask(match_points[x], IMG_WDT, IMG_HT), elems = tf.range(0, batch_size, 1), dtype=tf.float32)
velo1_pooled = tf.nn.max_pool(velo1/80.0, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
velo2_pooled = tf.nn.max_pool(velo2/80.0, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
velo3_pooled = tf.nn.max_pool(velo3/80.0, ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
predicted_vector = global_agg_net.End_Net_Out(cam1/255.0, cam2/255.0, cam3/255.0, velo1_pooled, velo2_pooled, velo3_pooled, False, True, keep_prob)
predicted_pose = tf.map_fn(lambda x: exponential_map_single(predicted_vector[x]), elems = tf.range(0, batch_size, 1), dtype=tf.float32)

expected_pose = tf.map_fn(lambda x: exponential_map_single(expected_vector[x]), elems = tf.range(0, batch_size, 1), dtype=tf.float32)

# 由激光雷达与相机的校准矩阵计算相机间位姿
with tf.variable_scope('compute_cam_pose'):
    velo2cam_mat = tf.matmul(tf.matmul(predicted_pose, decalib_mat), velo_to_cams)
    cam2velo_mat = tf.matrix_inverse(velo2cam_mat)
    pre_cam_mat = tf.matmul(cam2velo_mat, tf.matmul(velo_pose, velo2cam_mat)) # b*4*4

    velo2cam_mat = tf.matmul(tf.matmul(expected_pose, decalib_mat), velo_to_cams)
    cam2velo_mat = tf.matrix_inverse(velo2cam_mat)
    exp_cam_mat = tf.matmul(cam2velo_mat, tf.matmul(velo_pose, velo2cam_mat)) # b*4*4
with tf.variable_scope('3D_project'): #
    pre_depth, pre_proj_coords, cloud_pred = tf.map_fn(lambda x:at._simple_transformer(velo2[x,:,:,0], predicted_pose[x], pre_cam_mat[x], p_rect0[x]),
                                                       elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32, tf.float32))

    exp_depth, exp_proj_coords, cloud_exp = tf.map_fn(lambda x:at._simple_transformer(velo2[x,:,:,0], expected_pose[x], exp_cam_mat[x], p_rect0[x]),
                                                   elems = tf.range(0, batch_size, 1), dtype = (tf.float32, tf.float32, tf.float32))

with tf.variable_scope('compute_losses'):
    # cloud_loss = model_utils.get_emd_loss(cloud_pred, cloud_exp)
    cloud_exp = tf.reshape(cloud_exp, [-1,3])
    cloud_pred = tf.reshape(cloud_pred, [-1,3])
    dist_error = cloud_pred - cloud_exp
    cloud_loss = tf.reduce_mean(tf.norm(dist_error, axis=1)) * 0.01

    proj_img = bilinear_sampler(cam3, pre_proj_coords)
    exp_proj_img = bilinear_sampler(cam3, exp_proj_coords)

    zeros_target = tf.zeros_like(exp_depth)
    depth_mask = tf.not_equal(exp_depth, zeros_target)  # 将ZZ中不为0的元素置1
    depth_mask = tf.cast(depth_mask, 'float32')
    pre_depth_mask = tf.not_equal(pre_depth, zeros_target)  # 将ZZ中不为0的元素置1
    pre_depth_mask = tf.cast(pre_depth_mask, 'float32')

    true_proj_error = tf.reduce_mean(tf.abs(exp_proj_img[:,:,:,:] - cam2[:,:,:,:]), 3)
    weight_mask = (1 - tf.abs(true_proj_error) / 255) ** 2 # tf.divide(ones, tf.abs(true_proj_error) + 0.1)
    sift_match_mask = tf.multiply(weight_mask, sift_match_mask) * depth_mask[:,:,:,0]
    nonzero_num = tf.count_nonzero(sift_match_mask)
    nonzero_num = tf.cast(nonzero_num, 'float32') + 1e-6
    curr_proj_error = tf.reduce_mean(tf.abs(proj_img[:,:,:,:] - cam2[:,:,:,:]), 3)


    photometric_loss1 = cloud_loss # + depth_dist # + depth_error
    photometric_loss2 = tf.reduce_sum(curr_proj_error/255 * sift_match_mask)/ nonzero_num # 20190919
#  tf.nn.l2_loss(curr_proj_error/255 * sift_match_mask) * 0.1

    trans, rot = compute_pose_loss(expected_vector, predicted_vector)
    trans_loss = tf.nn.l2_loss(trans)
    rot_loss = tf.nn.l2_loss(rot)

    vector_loss = rot_loss + trans_loss

    pre_match_error = compute_match_loss(match_points, intrinsics, pre_cam_mat)
    exp_match_error = compute_match_loss(match_points, intrinsics, exp_cam_mat)
    w = (1 - exp_match_error) ** 2 # 1 / (exp_match_error + 1e-6)
    match_loss = tf.nn.l2_loss(pre_match_error * w) * 0.01
        # tf.nn.l2_loss(pre_match_error * w) * 0.01 20190919
    # tf.nn.l2_loss(tf.abs(pre_match_error - exp_match_error) * w) * 20

    tv = tf.trainable_variables() # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
    reg_loss = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) # 0.001是lambda超参数
    origin_loss = photometric_loss1  + photometric_loss2 + vector_loss + match_loss
    validation_loss = vector_loss + photometric_loss1 + photometric_loss2 + reg_loss + match_loss #
    train_loss = origin_loss + reg_loss

global_step = tf.Variable(0, trainable=False) # 11250
learning_rate = tf.train.exponential_decay(learning_rate=config.net_params['learning_rate'], global_step=global_step, decay_steps = 30000, # 30000# 82500
                                           decay_rate=0.5, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss, global_step=global_step)

training_summary_1 = tf.summary.scalar('train_loss', train_loss)
training_summary_2 = tf.summary.scalar('rotation_error', rot_loss)
training_summary_3 = tf.summary.scalar('translation_error', trans_loss)
training_summary_4 = tf.summary.scalar('pixel_loss2', photometric_loss2)
# training_summary_5 = tf.summary.scalar('depth_error', depth_error)
training_summary_5 = tf.summary.scalar('learning_rate', learning_rate)
training_summary_6 = tf.summary.scalar('match_loss', match_loss)
# training_summary_7 = tf.summary.scalar('depth_dist', angle_error_log)
training_summary_8 = tf.summary.scalar('cloud_loss', cloud_loss)
# training_summary_9 = tf.summary.scalar('depth_dist', trans_error_log)

validation_summary_1 = tf.summary.scalar('Validation_loss', validation_loss)
validation_summary_2 = tf.summary.scalar('val_rotation_error', rot_loss)
validation_summary_3 = tf.summary.scalar('val_translation_error', trans_loss)
validation_summary_4 = tf.summary.scalar('val_pixel_loss2', photometric_loss2)
# validation_summary_5 = tf.summary.scalar('depth_error', depth_error)
validation_summary_6 = tf.summary.scalar('val_match_loss', match_loss)
# validation_summary_7 = tf.summary.scalar('val_depth_dist', depth_dist)
validation_summary_8 = tf.summary.scalar('val_cloud_loss', cloud_loss)
merge_train = tf.summary.merge([training_summary_1] + [training_summary_2]+[training_summary_3]+[training_summary_4]
                               +[training_summary_5]+ [training_summary_8] + [training_summary_6])
merge_val = tf.summary.merge([validation_summary_1]+[validation_summary_2]+[validation_summary_3]+[validation_summary_4]
                               + [validation_summary_8] + [validation_summary_6])


pre_points_2d = get_points2d(pre_depth, x)
exp_points_2d = get_points2d(exp_depth, x)

checkpoint_path = config.paths['checkpoint_path']
total_train = config.net_params['total_train']
total_val = config.net_params['total_validation']
load_batch = config.net_params['load_batch']
training_imgs_path = config.paths['training_imgs']
validation_imgs_path = config.paths['validation_imgs']
total_partitions_train = int(total_train / (load_batch * batch_size))
total_partitions_val = int(total_val / (load_batch * batch_size))
train_iterations = 0
validation_iterations = 0
saver = tf.train.Saver(max_to_keep = 100)
args = arg_parser()
config_tf = sess_config(args)
with tf.Session(config = config_tf) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs_simple_transformer/")
    if current_epoch == 0:
        writer.add_graph(sess.graph)
        sess.run(tf.assign(global_step, 0))
    if current_epoch > 0:
        print("Restoring Checkpoint")
        saver.restore(sess, checkpoint_path + "/model-%d" % current_epoch)
        print('global step:', sess.run(global_step), 'learning_rate:', sess.run(learning_rate))
        current_epoch += 1
        train_iterations = current_epoch * config.net_params['total_train'] / batch_size  # 训练总数24000
        validation_iterations = current_epoch * config.net_params['total_validation'] / batch_size
    for epoch in range(current_epoch,n_epochs):#
        dl.shuffle(1)
        for part in range(total_partitions_train):
            cam1_container, cam2_container, cam3_container, depth1_container, depth2_container, depth3_container, \
            expected_transform_container, velo_pose_container, match_points_container, decalib_mat_container, p_rects, trs = dl.load(part, mode = "train")

            for img1, img2, img3, depth_img1, depth_img2, depth_img3, exp_vector, velo_mat, pair_matches, decalib, p_rect, tr in zip(cam1_container, cam2_container,
                    cam3_container, depth1_container, depth2_container, depth3_container, expected_transform_container,
                    velo_pose_container, match_points_container,decalib_mat_container, p_rects, trs):
                random_disp = np.random.randint(batch_size)
                outputs = sess.run([expected_pose, pre_points_2d, predicted_pose, proj_img, exp_proj_img,
                                    train_loss, sift_match_mask, photometric_loss2, photometric_loss1, vector_loss,
                                     train_step, merge_train, exp_points_2d],
                                   feed_dict={cam1: img1, cam2: img2, cam3: img3, velo1: depth_img1, velo2: depth_img2,
                                              velo3: depth_img3, decalib_mat: decalib, keep_prob: 0.5, velo_to_cams:tr,
                                              p_rect0:p_rect, intrinsics:p_rect[:,:3,:3],
                               expected_vector: exp_vector, x: random_disp, velo_pose: velo_mat, match_points: pair_matches})

                if train_iterations % 50 == 0:
                    pre_pose0 = outputs[2]
                    exp_pose0 = outputs[0]

                    error_pose = np.matmul(pre_pose0[random_disp], np.linalg.inv(exp_pose0[random_disp]))
                    trans_error = abs(error_pose[:3, 3]).reshape(1, 3)
                    angle_error = ((abs(mat_to_angle(error_pose[:3, :3])) / 3.14) * 180).reshape(1, 3)

                    print('epoch: ', epoch, 'global_step:', train_iterations, 'loss:', outputs[5],
                          'trans_error:', trans_error, 'angle_error:', angle_error)

                    print('ph2_loss:',outputs[7],'ph1_loss:',outputs[8],'vector_loss:', outputs[9])
                    # print('loss weight:', outputs[15], 'bias:', outputs[16])
                if train_iterations % 1000 == 0:
                    dep_rgbimg = transform_function.points2d_to_rgbimg(outputs[1], img2[random_disp])
                    cv2.imwrite('../training_imgs/depth_rgbimg_' + str(train_iterations) + '.png', dep_rgbimg)

                if train_iterations % 1000 == 0:
                    # print('predicted trans vector:', outputs[0][random_disp, :3, 3], 'predicted angle:',
                    #       (angle_pre / 3.14) * 180)
                    exp_dep_rgbimg = transform_function.points2d_to_rgbimg(outputs[12], img2[random_disp])
                    cv2.imwrite('../training_imgs/exp_depth_rgbimg_' + str(train_iterations) + '.png', exp_dep_rgbimg)

                if train_iterations % 100 == 0:
                    writer.add_summary(outputs[11], train_iterations / 100)

                train_iterations += 1
        if epoch % 10 == 0:
            print("Saving after epoch %d" % epoch)
            saver.save(sess, checkpoint_path + "/model-%d" % epoch)
        trans_errors = []
        rot_errors = []
        for part in range(total_partitions_val):
            cam1_container, cam2_container, cam3_container, depth1_container, depth2_container, depth3_container, \
            expected_transform_container, velo_pose_container, match_points_container, decalib_mat_container, p_rects, trs= dl.load(part, mode="validation")
            for img1, img2, img3, depth_img1, depth_img2, depth_img3, exp_vector, velo_mat, pair_matches, decalib, p_rect, tr in zip(
                    cam1_container, cam2_container,
                    cam3_container, depth1_container, depth2_container, depth3_container, expected_transform_container,
                    velo_pose_container, match_points_container, decalib_mat_container, p_rects, trs):
                outputs = sess.run([pre_points_2d, predicted_pose, proj_img, validation_loss,
                                    vector_loss, photometric_loss2, merge_val, expected_pose],
                                   feed_dict={cam1: img1, cam2: img2, cam3: img3, velo1: depth_img1, velo2: depth_img2,
                                              velo3: depth_img3, decalib_mat: decalib, velo_to_cams:tr,
                                              p_rect0:p_rect, intrinsics:p_rect[:,:3,:3],
                                              expected_vector: exp_vector, x: random_disp, velo_pose: velo_mat,
                                              match_points: pair_matches, keep_prob:1.0})
                pre_pose0 = outputs[1]
                exp_pose0 = outputs[7]

                error_pose = np.matmul(pre_pose0[random_disp], np.linalg.inv(exp_pose0[random_disp]))
                trans_error = abs(error_pose[:3, 3]).reshape(1, 3)
                angle_error = ((abs(mat_to_angle(error_pose[:3, :3])) / 3.14) * 180).reshape(1, 3)

                angle_error = angle_error.reshape(1,3)
                batch_trans_error = np.mean(trans_error, 0)
                batch_angle_error = np.mean(angle_error, 0)
                rot_errors.append(batch_angle_error)
                trans_errors.append(batch_trans_error)
                print('epoch: ', epoch, 'validation_step:', validation_iterations, 'loss:', outputs[3], 'trans_error:', trans_error,
                      'angle_error:', angle_error)
                print('vector_loss:', outputs[4], 'ph2_loss:', outputs[5])
                if validation_iterations % total_partitions_val == 0:
                    dep_rgbimg = transform_function.points2d_to_rgbimg(outputs[0], img2[random_disp])
                    cv2.imwrite(validation_imgs_path + "/validate_depth_%d.png" % validation_iterations, dep_rgbimg)

                    writer.add_summary(outputs[6], validation_iterations/total_partitions_val)
                validation_iterations += 1
        tr = np.mean(trans_errors, 0)
        rot = np.mean(rot_errors, 0)
        print('ode_epoch: ', epoch, 'trans error:', tr, 'angle error:', rot)
        trans_errors = []
        rot_errors = []


