import tensorflow as tf
import numpy as np


import config_res as config

IMG_HT = config.depth_img_params['IMG_HT']
IMG_WDT = config.depth_img_params['IMG_WDT']


shape = (IMG_HT, IMG_WDT)


def to_grid_xy(grids_flatten):
    """
    数据格式转换
    :param grids_flatten: x,y坐标, [ht*wd, 2]
    :return: 返回与图片大小一致的坐标数据，[ht, wd, 2]
    """
    grid_x = tf.reshape(grids_flatten[:, 0], (IMG_HT, IMG_WDT))
    grid_y = tf.reshape(grids_flatten[:, 1], (IMG_HT, IMG_WDT))
    grid_x = tf.expand_dims(grid_x, -1)
    grid_y = tf.expand_dims(grid_y, -1)
    grid = tf.concat([grid_x, grid_y], axis=2)
    return grid


def _simple_transformer(depth_map, calib_pose, cam_pose, p_rect): #, cam_pose
    """
    根据参数进行三维转换
    :param depth_map: 错误校准的深度图， [ht, wd]
    :param calib_pose: 输入深度图的校准参数， 【4, 4】
    :param cam_pose: 前一帧到后一帧的相机间位姿，【4, 4】
    :param p_rect: 相机内参, [4, 4]
    :return: calib_depth_map 校准后的深度图【ht, wd, 1】； grids_depth 校准后深度点的坐标【ht, wd, 2】；
             grids_wrap 根据深度图和相机间位姿，得到后一帧图像在前一帧图像坐标系下的投影坐标 【ht, wd, 2】;
             transformed_depth_map 转换后每个深度点的深度值【ht, wd】
    """
    batch_grids_depth, transformed_depth_map, sparse_cloud = _3D_meshgrid_batchwise_diff(depth_map, calib_pose, p_rect)
    calib_depth_map = _bilinear_sampling(transformed_depth_map, batch_grids_depth)
    # calib_depth_map, sparse_cloud = _3D_transform(depth_map, calib_pose, p_rect)
    batch_grids_wrap = project_coords(calib_depth_map, cam_pose, p_rect)
    grids_wrap = to_grid_xy(batch_grids_wrap)
    # grids_depth = to_grid_xy(batch_grids_depth)

    return calib_depth_map , grids_wrap, sparse_cloud # , grids_depth


def calib_depth(depth_map, calib_pose, p_rect):
    """
  在迭代校准时使用
    """
    batch_grids_depth, transformed_depth_map, _ = _3D_meshgrid_batchwise_diff(depth_map, calib_pose, p_rect)

    calib_depth_map = _bilinear_sampling(transformed_depth_map, batch_grids_depth)

    return calib_depth_map


def test_calib_depth(depth_map, calib_pose, p_rect):
    """
    根据参数进行三维转换
    :param depth_map: 错误校准的深度图， [ht, wd]
    :param calib_pose: 输入深度图的校准参数， 【4, 4】
    :param cam_pose: 前一帧到后一帧的相机间位姿，【4, 4】
    :param p_rect: 相机内参, [4, 4]
    :return: calib_depth_map 校准后的深度图【ht, wd, 1】； grids_depth 校准后深度点的坐标【ht, wd, 2】；
             grids_wrap 根据深度图和相机间位姿，得到后一帧图像在前一帧图像坐标系下的投影坐标 【ht, wd, 2】;
             transformed_depth_map 转换后每个深度点的深度值【ht, wd】
    """
    p_rect_ni = tf.matrix_inverse(p_rect)
    batch_grids_depth, transformed_depth_map, _ = _3D_meshgrid_batchwise_diff(depth_map, calib_pose, p_rect)

    calib_depth_map = _bilinear_sampling(transformed_depth_map, batch_grids_depth)

    return calib_depth_map


def points_transform(de_points_2d, calib_pose, p_rect):

    """
    Creates 3d sampling meshgrid
    """
    zz = tf.reshape(de_points_2d[:, 2], [1, -1])
    xy = tf.reshape(de_points_2d[:, 0:2], [2, -1])
    ones = tf.ones_like(zz)

    sampling_grid_2d = tf.concat([xy, ones], 0)

    projection_grid_3d = sampling_grid_2d * zz

    homog_points_3d = tf.concat([projection_grid_3d, ones], 0)  # [4,N]
    points_cam = tf.matmul(tf.matrix_inverse(p_rect), homog_points_3d)
    points_2d = tf.matmul(p_rect, tf.matmul(calib_pose, points_cam))
    Z = points_2d[2, :]

    x = tf.transpose(points_2d[0, :] / Z)

    y = tf.transpose(points_2d[1, :] / Z)

    ca_points_2d = tf.stack([x, y, Z], 1)

    return ca_points_2d


def _3D_transform(de_points_2d, calib_pose, p_rect):

    """
    Creates 3d sampling meshgrid
    """
    zz = tf.reshape(de_points_2d[:, 2], [1, -1])
    xy = tf.reshape(de_points_2d[:, 0:2], [2, -1])
    ones = tf.ones_like(zz)

    sampling_grid_2d = tf.concat([xy, ones], 0)

    projection_grid_3d = sampling_grid_2d * zz

    homog_points_3d = tf.concat([projection_grid_3d, ones], 0)  # [4,N]
    points_cam = tf.matmul(tf.matrix_inverse(p_rect), homog_points_3d)
    points_2d = tf.matmul(p_rect, tf.matmul(calib_pose, points_cam))

    Z = points_2d[2, :]
    x_dash_pred = points_2d[0, :]
    y_dash_pred = points_2d[1, :]

    point_cloud = tf.stack([x_dash_pred, y_dash_pred, Z], 1)

    sparse_point_cloud = sparsify_cloud(point_cloud)

    x = tf.transpose(points_2d[0, :] / Z)
    x = tf.clip_by_value(x, 0.0, IMG_WDT - 1)
    y = tf.transpose(points_2d[1, :] / Z)
    y = tf.clip_by_value(y, 0.0, IMG_HT - 1)

    updated_indices = tf.expand_dims(tf.cast(y * IMG_WDT + x, 'int32'), 1)
    calib_depth_map = tf.scatter_nd(updated_indices, Z, tf.constant([IMG_WDT*IMG_HT]))
    calib_depth_map = tf.reshape(calib_depth_map, (IMG_HT, IMG_WDT))
    calib_depth_map = tf.expand_dims(calib_depth_map, -1)
    return calib_depth_map, sparse_point_cloud


def to_tarimg(depth, img):
    zeros_target = tf.zeros_like(depth)
    mask = tf.not_equal(depth, zeros_target)  # 将ZZ中不为0的元素置1
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 1, 3])
    mask = tf.cast(mask, 'float32')
    tarimg = img * mask
    return tarimg


def sparsify_cloud(S):

    """
    Cluster centers of point clouds used to sparsify cloud for Earth Mover's Distance. Using 4096 centroids
    """

    with tf.device('/cpu:0'):

        point_limit = 5000 # 4096
        no_points = tf.shape(S)[0]
        no_partitions = no_points/tf.constant(point_limit, dtype=tf.int32)
        no_partitions = tf.cast(no_partitions, dtype=tf.int32)
        saved_points = tf.gather_nd(S, [tf.expand_dims(tf.range(0, no_partitions*point_limit), 1)])
        saved_points = tf.reshape(saved_points, [point_limit, no_partitions, 3])
        saved_points_sparse = tf.reduce_mean(saved_points, 1)

        return saved_points_sparse


def project_coords(depth_img, cam_pose, p_rect):
    """

    :param depth_img:
    :param cam_pose:
    :param p_rect:
    :return:
    """
    height, width, _= depth_img.get_shape().as_list()
    x_index = tf.linspace(0.0, width, width)  # 等间隔生成-1到1的长为1242的数组
    y_index = tf.linspace(0.0, height, height)  # 等间隔生成-1到1的长为375的数组

    z_index = tf.range(0, width * height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten
    x_t_flat = tf.reshape(x_t, [1, -1])
    y_t_flat = tf.reshape(y_t, [1, -1])
    ZZ = tf.reshape(depth_img, [-1])

    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))

    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    projection_grid_3d = sampling_grid_2d_sparse * ZZ_saved

    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)  # [4,N]

    points_cam = tf.matmul(tf.matrix_inverse(p_rect), homog_points_3d)
    points_2d = tf.matmul(p_rect, tf.matmul(cam_pose, points_cam))
    Z = points_2d[2, :]

    x = tf.transpose(points_2d[0, :] / Z)
    y = tf.transpose(points_2d[1, :] / Z)

    mask_int = tf.cast(mask, 'int32')

    updated_indices = tf.expand_dims(tf.boolean_mask(mask_int * z_index, mask), 1)

    updated_x = tf.scatter_nd(updated_indices, x, tf.constant([width * height]))

    updated_y = tf.scatter_nd(updated_indices, y, tf.constant([width * height]))

    reprojected_grid = tf.stack([updated_x, updated_y], 1)


    return reprojected_grid


def _3D_meshgrid_batchwise_diff(depth_img, calib_pose, p_rect):

    """
    Creates 3d sampling meshgrid
    """
    height, width, = depth_img.get_shape().as_list()
    x_index = tf.linspace(0.0, width, width)
    y_index = tf.linspace(0.0, height, height)

    z_index = tf.range(0, width * height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten
    x_t_flat = (tf.reshape(x_t, [1, -1]))
    y_t_flat = (tf.reshape(y_t, [1, -1]))
    ZZ = tf.reshape(depth_img, [-1])

    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))
    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    projection_grid_3d = sampling_grid_2d_sparse * ZZ_saved

    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)  # [4,N]
    # points_cam = tf.matmul(p_rect_radni, homog_points_3d)
    points_cam = tf.matmul(tf.matrix_inverse(p_rect), homog_points_3d)
    points_2d = tf.matmul(p_rect, tf.matmul(calib_pose, points_cam))
    Z = points_2d[2, :]
    x_dash_pred = points_2d[0, :]
    y_dash_pred = points_2d[1, :]

    point_cloud = tf.stack([x_dash_pred, y_dash_pred, Z], 1)

    sparse_point_cloud = sparsify_cloud(point_cloud)

    x = tf.transpose(points_2d[0, :] / Z)
    # x = tf.clip_by_value(x, 0.0, IMG_WDT - 1)
    y = tf.transpose(points_2d[1, :] / Z)
    # y = tf.clip_by_value(y, 0.0, IMG_HT - 1)

    mask_int = tf.cast(mask, 'int32')

    updated_indices = tf.expand_dims(tf.boolean_mask(mask_int * z_index, mask), 1)

    updated_Z = tf.scatter_nd(updated_indices, Z, tf.constant([width * height]))
    updated_x = tf.scatter_nd(updated_indices, x, tf.constant([width * height]))

    updated_y = tf.scatter_nd(updated_indices, y, tf.constant([width * height]))

    reprojected_grid = tf.stack([updated_x, updated_y], 1)

    transformed_depth = tf.reshape(updated_Z, (IMG_HT, IMG_WDT))

    return reprojected_grid, transformed_depth, sparse_point_cloud


def reverse_all(z):

    """Reversing from cantor function indices to correct indices"""

    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8.*z + 1.) - 1.)/2.0)
    t = (w**2 + w)/2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, IMG_HT - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:,0], 1), 0.0, IMG_WDT - 1)

    return tf.concat([y,x], 1)


def get_pixel_value(img, x, y):

    """Cantor pairing for removing non-unique updates and indices. At the time of implementation, unfixed issue with scatter_nd causes problems with int32 update values. Till resolution, implemented on cpu """

    with tf.device('/cpu:0'):
        indices = tf.stack([y, x], 2)
        indices = tf.reshape(indices, (IMG_HT*IMG_WDT, 2))
        values = tf.reshape(img, [-1])

        Y = indices[:,0]
        X = indices[:,1]
        s = (X + Y)*(X + Y + 1)
        Z = tf.cast(s/2, 'int32') + Y

        filtered, idx = tf.unique(tf.squeeze(Z))
        updated_values  = tf.unsorted_segment_max(values, idx, tf.shape(filtered)[0])

        # updated_indices = tf.map_fn(fn=lambda i: reverse(i), elems=filtered, dtype=tf.float32)
        updated_indices = reverse_all(filtered)
        updated_indices = tf.cast(updated_indices, 'int32')
        resolved_map = tf.scatter_nd(updated_indices, updated_values, img.shape)

        return resolved_map

def _bilinear_sampling(img, batch_grids_depth):

    """
    Sampling from input image and performing bilinear interpolation
    """
    x_func = tf.reshape(batch_grids_depth[:,0], (IMG_HT, IMG_WDT))
    y_func = tf.reshape(batch_grids_depth[:,1], (IMG_HT, IMG_WDT))
    max_y = tf.constant(IMG_HT - 1, dtype=tf.int32)
    max_x = tf.constant(IMG_WDT - 1, dtype=tf.int32)

    x = tf.clip_by_value(x_func, 0.0, tf.cast(max_x, 'float32'))
    y = tf.clip_by_value(y_func, 0.0, tf.cast(max_y, 'float32'))

    x_round = tf.round(x)
    y_round = tf.round(y)
    x_round = tf.cast(x_round,'int32')
    y_round = tf.cast(y_round, 'int32')
    loc = get_pixel_value(img,x_round,y_round)
    loc = tf.reshape(loc, (IMG_HT, IMG_WDT, 1))


    return loc
