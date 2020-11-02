import tensorflow as tf
import tensorflow.contrib.slim as slim
import config_res as config

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']


def compute_pose_loss(prior_pose_vec, pred_pose_vec):
    """
    计算6自由度校准参数的线性回归误差，旋转向量和平移向量各自与ground truth的欧式距离
    :param prior_pose_vec:ground truth校准参数，[bs, 6]
    :param pred_pose_vec:预测的校准参数， [bs, 6]
    :return:平移向量的欧氏距离误差，[bs, 1]; 旋转向量的欧氏距离误差，[bs, 1]
    """
    rot_vec_err = tf.norm(prior_pose_vec[:, 3:] - pred_pose_vec[:, 3:], axis=1)
    trans_err = tf.norm(prior_pose_vec[:, :3] - pred_pose_vec[:, :3], axis=1)

    return trans_err, rot_vec_err


def compute_match_loss(matches, intrinsics, cam_mat):
    """
    利用对极几何约束，计算匹配的sift特征点对在输入的相邻帧相机位姿下的特征点到极线的距离
    :param matches:两帧图片匹配的sift特征点对，[bs, match_points_num, 2]
    :param intrinsics:相机内参， [bs, 3, 3]
    :param cam_mat:前一帧到后一帧的相机位姿， [bs, 3, 3]
    :return:距离和
    """
    # batch_size, match_num, _ = matches.get_shape().as_list()
    batch_size = tf.shape(matches)[0]
    match_num = tf.shape(matches)[1]
    points1 = tf.slice(matches, [0, 0, 0], [-1, -1, 2])
    points2 = tf.slice(matches, [0, 0, 2], [-1, -1, 2])
    ones = tf.ones([batch_size, match_num, 1])
    points1 = tf.concat([points1, ones], axis=2)
    points2 = tf.concat([points2, ones], axis=2)

    tr_vec = tf.slice(cam_mat, [0, 0, 3], [-1, 3, 1]) # bs*3*1
    translation_ssm = skew_symmetric_mat3(tr_vec)
    #旋转变换矩阵
    rot_mat = tf.slice(cam_mat, [0, 0, 0], [-1, 3, 3])
    # R12[t12]
    essential_mat = tf.matmul(rot_mat, translation_ssm)
    intrinsics_inv = tf.matrix_inverse(intrinsics)
    fmat = tf.matmul(tf.transpose(intrinsics_inv, [0, 2, 1]), essential_mat)
    fmat = tf.matmul(fmat, intrinsics_inv)
    fmat = tf.expand_dims(fmat, axis=1)
    fmat_tiles = tf.tile(fmat, [1, match_num, 1, 1])
    epi_lines = tf.matmul(fmat_tiles, tf.expand_dims(points1, axis=3))
    dist_p2l = tf.abs(tf.matmul(tf.transpose(epi_lines, perm=[0, 1, 3, 2]), tf.expand_dims(points2, axis=3)))

    a = tf.slice(epi_lines, [0,0,0,0], [-1,-1,1,-1])
    b = tf.slice(epi_lines, [0,0,1,0], [-1,-1,1,-1])
    dist_div = tf.sqrt(a*a + b*b) + 1e-6
    dist_p2l = tf.reduce_mean(dist_p2l / dist_div)

    return dist_p2l


def depth_coords_dist(exp_coords, pre_coords):
    """
    计算由预测参数转换的深度图点和ground truth参数转换的深度图点之间的坐标距离
    :param exp_coords:对于错误校准的深度图，用实际的校准参数进行转换，得到的每一个深度点实际该投影的坐标，[bs, height, width, 2]
    :param pre_coords:类似，用预测的校准参数，[bs, height, width, 2]
    :return: 欧式距离，平方和
    """
    xe, ye = tf.split(exp_coords, [1, 1], axis=3)
    xe = tf.cast(xe/IMG_WDT, 'float32')
    ye = tf.cast(ye/IMG_HT, 'float32')

    xp, yp = tf.split(pre_coords, [1, 1], axis=3)
    xp = tf.cast(xp/IMG_WDT, 'float32')
    yp = tf.cast(yp/IMG_HT, 'float32')

    xx = tf.reshape(xe - xp, [-1, 1])
    yy = tf.reshape(ye - yp, [-1, 1])
    xy_d = tf.concat([xx, yy], 1)
    dist = tf.norm(xy_d, axis=1)
    dist = tf.nn.l2_loss(dist)

    return dist


def skew_symmetric_mat3(vec3):
    """compute the skew symmetric matrix for cross product

    Arguments:
        vec {vector of shape [batch_size, 3, 1]}
    """
    bs = tf.shape(vec3)[0]  # batch size
    a1 = tf.slice(vec3, [0, 0, 0], [-1, 1, -1])
    a2 = tf.slice(vec3, [0, 1, 0], [-1, 1, -1])
    a3 = tf.slice(vec3, [0, 2, 0], [-1, 1, -1])
    zeros = tf.zeros([bs, 1, 1])
    row1 = tf.concat([zeros, -a3, a2], axis=2)
    row2 = tf.concat([a3, zeros, -a1], axis=2)
    row3 = tf.concat([-a2, a1, zeros], axis=2)
    vec3_ssm = tf.concat([row1, row2, row3], axis=1)
    return vec3_ssm
