import tensorflow as tf
import numpy as np
import cv2
import os
import math
import config_res as config


dataset = np.loadtxt(config.paths['train_data'], dtype = str)
total = config.net_params['total']
IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']
batch_size = config.net_params['batch_size']
cam_img_path = config.paths['cam_img']
points_pair_path = config.paths['match_point']

load_batch = config.net_params['load_batch']

total = config.net_params['total']
total_train_frame = config.net_params['total_train']
total_test = config.net_params['total_test']
# total_valdition = config.net_params['total_validation']
depth_img_path = config.paths['transformed_depth']
partition_limit = load_batch * batch_size


dataset_train = dataset #[:total_train_frame] #
dataset_test = np.loadtxt(config.paths['test_data'], dtype = str)
dataset_validation = dataset_test[:100]# [total_train_frame:] # _train_frame

# 00 01 02
tr0 = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03,-1.198459927713e-02,
                -7.210626507497e-03, 8.081198471645e-03,-9.999413164504e-01, -5.403984729748e-02,
                9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
                0.0, 0.0, 0.0, 1.0], 'float32').reshape(4,4)
# 03
tr3 = np.array([2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03,
                1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02,
                9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-01,
                0.0, 0.0, 0.0, 1.0], 'float32').reshape(4,4)

# 04 05 06
tr4 = np.array([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                0.0, 0.0, 0.0, 1.0], 'float32').reshape(4,4)

def int2str(x):
    s = str(x)
    l = len(s)
    if l < 6:
        b = 6 - l
        for i in range(b):
            s = '0' + str(s)
    return s


def shuffle(s):
    if s == 1:
        # dataset_train = np.random.sample(dataset_train, 6000)
        np.random.shuffle(dataset_train)
        np.random.shuffle(dataset_validation)
    else:
        np.random.shuffle(dataset_test)


def mat_to_angle(r):
    t = np.sqrt(r[2, 0] ** 2 + r[2, 2] ** 2)
    angle_z = math.atan2(r[1, 0], r[0, 0])
    angle_y = math.atan2(-1 * r[2, 0], t)
    angle_x = math.atan2(r[2, 1], r[2, 2])
    return np.array([angle_x, angle_y, angle_z])


def load(p_no, mode):
    if (mode == "train"):

        dataset_part = dataset_train[p_no * partition_limit:(p_no + 1) * partition_limit]
    elif (mode == "validation"):

        dataset_part = dataset_validation[p_no * partition_limit:(p_no + 1) * partition_limit]
    elif (mode == "test"):

        dataset_part = dataset_test[p_no * partition_limit:(p_no + 1) * partition_limit]

    cam1_name = dataset_part[:, 0]
    decalib_mats = np.float32(dataset_part[:, 1: 17])
    cam_poses = np.float32(dataset_part[:, 33: 49])
    p_rects = np.float32(dataset_part[:, 17: 33])
    scales = np.float32(dataset_part[:, 49])


    cam1_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype=np.float32)
    cam2_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype=np.float32)
    cam3_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 3), dtype=np.float32)
    depth1_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype=np.float32)
    depth2_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype=np.float32)
    depth3_container = np.zeros((partition_limit, IMG_HT, IMG_WDT, 1), dtype=np.float32)
    cam_pose_container = np.zeros((partition_limit, 4, 4), dtype=np.float32)
    expected_transform_container = np.zeros((partition_limit, 6), dtype=np.float32)
    velo_pose_container = np.zeros((partition_limit, 4, 4), dtype=np.float32)
    decalib_pose_container = np.zeros((partition_limit, 4, 4), dtype=np.float32)
    match_points_container = np.zeros((partition_limit, 100, 4), dtype=np.float32)
    p_rect_scale_container = np.zeros((partition_limit, 4, 4), dtype=np.float32)
    tr_container = np.zeros((partition_limit, 4, 4), dtype=np.float32)
    vector_t = np.zeros((3, 1), dtype=np.float32)
    c_idx = 0
    for cam1, decalib_mat, cam_pose, p_rect, scale in zip(cam1_name, decalib_mats,  cam_poses, p_rects, scales):

        s_name = cam1.split('/')
        fn = s_name[0] + '/image_2/'
        s = s_name[-1].split('.')[0]
        seq = s.split('_')[0]
        nn = s.split('_')[1]
        seq2 = int2str(int(seq) + 1)
        seq3 = int2str(int(seq) + 2)
        match_points = np.loadtxt(os.path.join(points_pair_path, s_name[0], seq2 + '.txt'))
        match_points_container[c_idx, :, :] = match_points[:100]
        file_seq = s_name[0]
        if file_seq == '00' or file_seq == '01' or file_seq == '02':
            velo_to_cam = tr0
        elif file_seq == '03':
            velo_to_cam = tr3
        elif file_seq == '04' or file_seq == '05' or file_seq == '06':
            velo_to_cam = tr4

        cam1_path = fn + seq + '.png'
        # print(cam1_path)
        img1 = np.float32(cv2.imread(os.path.join(cam_img_path,cam1_path)))
        ht = img1.shape[0]
        wdt = img1.shape[1]
        wd_s = int(round(wdt * scale))
        ht_s = int(round(ht * scale))
        img1[0:5, :, :] = 0.0; img1[:, 0:5, :] = 0.0; img1[IMG_HT - 5:, :, :] = 0.0; img1[:, IMG_WDT - 5:, :] = 0.0;
        scaled_img1 = cv2.resize(img1,(wd_s, ht_s))
        cam1_container[c_idx, :, :, :] = scaled_img1[:IMG_HT, :IMG_WDT, :] # cv2.resize(img1,(IMG_WDT, IMG_HT))


        cam2_path = fn + seq2 + '.png'
        img2 = np.float32(cv2.imread(os.path.join(cam_img_path,cam2_path)))
        img2[0:5, :, :] = 0.0; img2[:, 0:5, :] = 0.0; img2[IMG_HT - 5:, :, :] = 0.0; img2[:, IMG_WDT - 5:, :] = 0.0;
        # cam2_container[c_idx, :, :, :] = cv2.resize(img2,(IMG_WDT, IMG_HT))
        scaled_img2 = cv2.resize(img2, (wd_s, ht_s))
        cam2_container[c_idx, :, :, :] = scaled_img2[:IMG_HT, :IMG_WDT, :]


        cam3_path = fn + seq3 + '.png'
        img3 = np.float32(cv2.imread(os.path.join(cam_img_path, cam3_path)))
        img3[0:5, :, :] = 0.0; img3[:, 0:5, :] = 0.0; img3[IMG_HT - 5:, :, :] = 0.0; img3[:, IMG_WDT - 5:, :] = 0.0;
        # cam3_container[c_idx, :, :, :] = cv2.resize(img3,(IMG_WDT, IMG_HT))
        scaled_img3 = cv2.resize(img3, (wd_s, ht_s))
        cam3_container[c_idx, :, :, :] = scaled_img3[:IMG_HT, :IMG_WDT, :]


        #depth1_name = s + '_t1.png'
        depth1_name = s_name[0] + '/' + s + '_t1.png'
        depth1 = np.float32(cv2.imread(os.path.join(depth_img_path,depth1_name), cv2.IMREAD_GRAYSCALE))
        depth1[0:5, :] = 0.0; depth1[:, 0:5] = 0.0; depth1[IMG_HT - 5:, :] = 0.0; depth1[:, IMG_WDT - 5:] = 0.0;
        depth1_container[c_idx, :, :, 0] = depth1

       # depth2_name = s + '_t2.png'
        depth2_name = s_name[0] + '/' + s + '_t2.png'
        depth2 = np.float32(cv2.imread(os.path.join(depth_img_path, depth2_name), cv2.IMREAD_GRAYSCALE))
        depth2[0:5, :] = 0.0; depth2[:, 0:5] = 0.0; depth2[IMG_HT - 5:, :] = 0.0; depth2[:, IMG_WDT - 5:] = 0.0;
        depth2_container[c_idx, :, :, 0] = depth2

       # depth3_name = s + '_t3.png'
        depth3_name = s_name[0] + '/' + s + '_t3.png'
        depth3 = np.float32(cv2.imread(os.path.join(depth_img_path, depth3_name), cv2.IMREAD_GRAYSCALE))
        depth3[0:5, :] = 0.0; depth3[:, 0:5] = 0.0; depth3[IMG_HT - 5:, :] = 0.0; depth3[:, IMG_WDT - 5:] = 0.0;
        depth3_container[c_idx, :, :, 0] = depth3


        init_decalib_mat = decalib_mat.reshape(4, 4)
        decalib_pose_container[c_idx, :, :] = init_decalib_mat
        calib_mat = np.linalg.inv(init_decalib_mat)
        cam_to_velo = np.linalg.inv(velo_to_cam)
        cam_mat = cam_pose.reshape(4,4)
        velo_mat = np.matmul(velo_to_cam, np.matmul(cam_mat, cam_to_velo))

        r = calib_mat[:3,:3]
        vector_w, _ = cv2.Rodrigues(r)  # 3*1
        vector_t[:, 0] = calib_mat[:3, 3]
        vector = np.vstack((vector_t, vector_w)).reshape(1, 6)

        expected_transform_container[c_idx, :] = vector
        # cam_pose_container[c_idx, :, :] = cam_mat
        velo_pose_container[c_idx, :, :] = velo_mat
        # p_rect_scale = scale_p_rect(scale)
        p_rect_scale_container[c_idx, :, :] = p_rect.reshape(4,4)
        tr_container[c_idx, :, :] = velo_to_cam
        c_idx += 1
    cam1_container = cam1_container.reshape(load_batch, batch_size, IMG_HT, IMG_WDT, 3)
    cam2_container = cam2_container.reshape(load_batch, batch_size, IMG_HT, IMG_WDT, 3)
    cam3_container = cam3_container.reshape(load_batch, batch_size, IMG_HT, IMG_WDT, 3)
    depth1_container = depth1_container.reshape(load_batch, batch_size,  IMG_HT, IMG_WDT, 1)
    depth2_container = depth2_container.reshape(load_batch, batch_size,  IMG_HT, IMG_WDT, 1)
    depth3_container = depth3_container.reshape(load_batch, batch_size,  IMG_HT, IMG_WDT, 1)
    expected_transform_container = expected_transform_container.reshape(load_batch, batch_size, 6)
    # cam_pose_container = cam_pose_container.reshape(load_batch, batch_size, 4, 4)
    velo_pose_container = velo_pose_container.reshape(load_batch, batch_size, 4, 4)
    match_points_container = match_points_container.reshape(load_batch, batch_size, 100, 4)
    decalib_pose_container = decalib_pose_container.reshape(load_batch, batch_size, 4, 4)
    p_rect_scale_container = p_rect_scale_container.reshape(load_batch, batch_size, 4, 4)
    tr_container = tr_container.reshape(load_batch, batch_size, 4, 4)

    return cam1_container, cam2_container, cam3_container, depth1_container, depth2_container,depth3_container,\
            expected_transform_container, velo_pose_container, match_points_container, decalib_pose_container, \
           p_rect_scale_container, tr_container









