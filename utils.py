from __future__ import division
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math


def points2d_depimg(points_2d,width,height):
    Z = points_2d[2, :]
    x = tf.transpose(points_2d[0, :] / Z)
    y = tf.transpose(points_2d[1, :] / Z)
    # zero = tf.zeros([1], dtype='float32')
    x = tf.clip_by_value(x, 0, width - 1)
    y = tf.clip_by_value(y, 0, height - 1)
    x = tf.cast(x, 'int32')
    y = tf.cast(y, 'int32')
    xy = tf.transpose(tf.stack([y, x], 0))

    transformed_depth = tf.sparse_to_dense(xy, [height, width], Z, validate_indices=False)

    return transformed_depth

def decalib_depth(decalib, points, p_rect0, velo_to_cam, width, height):

    decalib_mat = pose_vec2mat(decalib)[0]
    trans_points_cam = tf.matmul(decalib_mat, (tf.matmul(velo_to_cam, tf.transpose(points))))
    points_2d = tf.matmul(p_rect0, trans_points_cam)
    Z = points_2d[2, :]
    x = tf.transpose(points_2d[0, :] / Z)
    y = tf.transpose(points_2d[1, :] / Z)
    # zero = tf.zeros([1], dtype='float32')
    x = tf.clip_by_value(x, 0, width-1)
    y = tf.clip_by_value(y, 0, height-1)
    x = tf.cast(x, 'int32')
    y = tf.cast(y, 'int32')
    xy = tf.transpose(tf.stack([y,x], 0))

    transformed_depth = tf.sparse_to_dense(xy,[height, width], Z, validate_indices=False)

    return transformed_depth


def get_points2d(calib_depth, x):

    calib_depth0 = calib_depth[x, :, :]
    height, width,_ = calib_depth0.get_shape().as_list()

    x_index = tf.linspace(0.0, width, width)  # 等间隔生成-1到1的长为1242的数组
    y_index = tf.linspace(0.0, height, height)  # 等间隔生成-1到1的长为375的数组

    x_t, y_t = tf.meshgrid(x_index, y_index)  # x_t是x_index重复375行，375*1242；y_t是y_index转置后重复1242列,375*1242

    # flatten
    x_t_flat = tf.reshape(x_t, [1,-1])  # [1,375*1242]
    y_t_flat = tf.reshape(y_t, [1,-1])  # [1,375*1242]
    ZZ = tf.reshape(calib_depth0, [-1])  # [1,375*1242]
    zeros_target = tf.zeros_like(ZZ)  # 将ZZ中所有元素值设为0
    mask = tf.not_equal(ZZ, zeros_target)  # 将ZZ中不为0的元素置1
    ones = tf.ones_like(x_t_flat)  # 将x_t_flat中所有元素值设为1

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)  # [3,375*1242]
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))  # 仅保留转置后sampling_grid_2d对应mask位置上为1的元素，再转置，
    ZZ_saved = tf.boolean_mask(ZZ, mask)  # 仅保留ZZ中不为0的元素

    points_2d = sampling_grid_2d_sparse*ZZ_saved

    return points_2d

def calib_points2d(decalib_depth, calib_pose, p_rect):

    height, width, _ = decalib_depth.get_shape().as_list()

    x_index = tf.linspace(0.0, width, width)  # 等间隔生成-1到1的长为1242的数组
    y_index = tf.linspace(0.0, height, height)  # 等间隔生成-1到1的长为375的数组

    x_t, y_t = tf.meshgrid(x_index, y_index)  # x_t是x_index重复375行，375*1242；y_t是y_index转置后重复1242列,375*1242

    # flatten
    x_t_flat = tf.reshape(x_t, [1, -1])  # [1,375*1242]
    y_t_flat = tf.reshape(y_t, [1, -1])  # [1,375*1242]
    ZZ = tf.reshape(decalib_depth, [-1])  # [1,375*1242]
    zeros_target = tf.zeros_like(ZZ)  # 将ZZ中所有元素值设为0
    mask = tf.not_equal(ZZ, zeros_target)  # 将ZZ中不为0的元素置1
    ones = tf.ones_like(x_t_flat)  # 将x_t_flat中所有元素值设为1

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)  # [3,375*1242]
    sampling_grid_2d_sparse = tf.transpose(
        tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))  # 仅保留转置后sampling_grid_2d对应mask位置上为1的元素，再转置，
    ZZ_saved = tf.boolean_mask(ZZ, mask)  # 仅保留ZZ中不为0的元素
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)
    projection_grid_3d = sampling_grid_2d_sparse * ZZ_saved

    homog_points_3d = tf.concat([projection_grid_3d, ones_saved], 0)  # [4,N]
    points_cam = tf.matmul(tf.matrix_inverse(p_rect), homog_points_3d)
    points_2d = tf.matmul(p_rect, tf.matmul(calib_pose, points_cam))

    return points_2d


def calib_depth(img, decalib_depth, calib_pose, p_rect):

    height, width, _ = decalib_depth.get_shape().as_list()
    img_pixel = tf.reshape(img,[height*width,3])

    points_2d = calib_points2d(decalib_depth,calib_pose,p_rect)

    Z = points_2d[2, :] # 开始

    x = tf.transpose(points_2d[0, :] / Z)
    y = tf.transpose(points_2d[1, :] / Z)

    # len = height * width - tf.shape(x)[0]
    # len = tf.cast(len, 'int32')
    # pad = [[0, len],[0, 0]]
    # points_2d_xy = tf.stack([x,y], 1)
    # points_2d_xy = tf.pad(points_2d_xy, pad)

    x = tf.clip_by_value(x, 0, width - 1)
    y = tf.clip_by_value(y, 0, height - 1)
    x = tf.cast(x, 'int32')
    y = tf.cast(y, 'int32')



    xy_idx = y * width + x
    updated_indices = tf.cast(xy_idx, 'int32')

    updated_indices = tf.expand_dims(updated_indices, 1)

    pixel = tf.gather_nd(img_pixel, updated_indices)[:,0]
    updated_Z = tf.scatter_nd(updated_indices, Z, tf.constant([width * height]))  # 根据indices将updates散布到新的（初始为零）张量
    updated_pixel = tf.scatter_nd(updated_indices, pixel, tf.constant([width * height]))
    transformed_depth = tf.reshape(updated_Z, (height, width))
    transformed_img = tf.reshape(updated_pixel, (height, width))

    # zeros_target = tf.zeros_like(transformed_depth)
    # mask = tf.not_equal(transformed_depth, zeros_target)  # 将ZZ中不为0的元素置1
    # mask = tf.cast(mask, 'float32')
    # transformed_img = img[:,:,0] * mask
    transformed_img = tf.reshape(transformed_img, (height, width))
    transformed_img = tf.expand_dims(transformed_img, -1)
    transformed_img = tf.tile(transformed_img, [1, 1, 3])
    transformed_img = tf.cast(transformed_img, 'float32')

    return transformed_depth, transformed_img



def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)
    # 注意旋转顺序Z-Y-X！！！
    rotMat = tf.matmul(tf.matmul(zmat, ymat), xmat)
    return rotMat

def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    # batch_size, _ = vec.get_shape().as_list()


    batch_size = tf.shape(vec)[0]
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]?? [batch,height, width]
      intrinsics: camera intrinsics [batch, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    batch = tf.shape(depth)[0]
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height*width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
      cam_coords: [batch, 4, height, width]
      proj: [batch, 4, 4]
    Returns:
      Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    batch = tf.shape(cam_coords)[0]
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)

    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])

    return coords


def to_tarimg(depth, img):
    zeros_target = tf.zeros_like(depth)
    mask = tf.not_equal(depth, zeros_target)  # 将ZZ中不为0的元素置1
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 1, 3])
    mask = tf.cast(mask, 'float32')
    tarimg = img * mask
    return tarimg



def projective_inverse_warp(img, depth, pose, intrinsics):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 4, 4], in the
            order of tx, ty, tz, rx, ry, rz
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    batch = tf.shape(img)[0]
    # Convert pose vector to matrix
    # pose = pose_vec2mat(pose)
    # Construct pixel grid coordinates
    # batch = 1
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel) # 合成视图转换后的坐标点
    output_img = bilinear_sampler(img, src_pixel_coords,depth)
    return output_img, cam_coords


def _repeat(x, n_repeats):
    rep = tf.transpose(
      tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

def bilinear_sampler(imgs, coords): #
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """

    with tf.name_scope('image_sampling'):
        # zeros_target = tf.zeros_like(depth)
        # mask = tf.not_equal(depth, zeros_target)  # 将ZZ中不为0的元素置1
        # mask = tf.expand_dims(mask, -1)
        # mask = tf.tile(mask, [1, 1, 1, 3])
        # mask = tf.cast(mask, 'float32')
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)

        inp_size = tf.shape(imgs)
        # coord_size = tf.shape(coords)
        out_size = tf.shape(coords) # coords.get_shape().as_list()
        # out_size[3] = tf.cast(inp_size[3],'float32')

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        # 相邻像素点坐标
        x0 = tf.floor(coords_x) # 向下取整
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')
        # 相邻像素点与原像素点的空间接近度
        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32') # img width
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32') # img width * height
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(out_size[0]), 'float32') * dim1,
                out_size[1] * out_size[2]), [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = tf.reshape(x0_safe + base_y1, [-1])
        idx10 = tf.reshape(x1_safe + base_y0, [-1])
        idx11 = tf.reshape(x1_safe + base_y1, [-1])
        # batch = tf.reshape(tf.range(out_size[0]),(-1,1))
        # base_per = tf.ones(shape=[out_size[0],dim1])
        # base = base_per * batch
        #
        # idx00 =  tf.reshape(x0_safe + base_y0, [-1])
        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), [out_size[0], out_size[1], out_size[2], inp_size[3]])
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), [out_size[0], out_size[1], out_size[2], inp_size[3]])
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), [out_size[0], out_size[1], out_size[2], inp_size[3]])
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), [out_size[0], out_size[1], out_size[2], inp_size[3]])

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        # mask = tf.add_n([w00, w01, w10, w11])
        # output = output * mask
        return output


def get_sift_match_mask(match_points, width, height):
    ones = tf.ones_like(match_points)[:, 0]
    x = match_points[:, 0]
    y = match_points[:, 1]
    xy_idx = y * width + x
    updated_indices = tf.cast(xy_idx, 'int32')
    updated_indices = tf.expand_dims(updated_indices, 1)
    mask = tf.scatter_nd(updated_indices, ones, tf.constant([width * height]))
    mask = tf.reshape(mask, (height, width))

    return mask

def get_reference_explain_mask(batch_size, img_height, img_width):

    tmp = np.array([0, 1])
    # 将[0,1]扩展为(4, , ,1)
    ref_exp_mask = np.tile(tmp, (batch_size, img_height, img_width, 1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask


def compute_exp_reg_loss(pred, ref):
    l = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(ref, [-1, 2]), logits=tf.reshape(pred, [-1, 2]))

    return tf.reduce_mean(l)

def exponential_map_single(vec):

    "Exponential Map Operation. Decoupled for SO(3) and translation t"

    with tf.name_scope("Exponential_map"):

        u = vec[:3] # 平移
        omega = vec[3:] # 旋转

        theta = tf.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])

        omega_cross = tf.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
        omega_cross = tf.reshape(omega_cross, [3,3])

        #Taylor's approximation for A,B and C not being used currently, approximations preferable for low values of theta

        # A = 1.0 - (tf.pow(theta,2)/factorial(3.0)) + (tf.pow(theta, 4)/factorial(5.0))
        # B = 1.0/factorial(2.0) - (tf.pow(theta,2)/factorial(4.0)) + (tf.pow(theta, 4)/factorial(6.0))
        # C = 1.0/factorial(3.0) - (tf.pow(theta,2)/factorial(5.0)) + (tf.pow(theta, 4)/factorial(7.0))

        A = tf.sin(theta)/theta

        B = (1.0 - tf.cos(theta))/(tf.pow(theta,2))

        C = (1.0 - A)/(tf.pow(theta,2))

        omega_cross_square = tf.matmul(omega_cross, omega_cross)

        R = tf.eye(3,3) + A*omega_cross + B*omega_cross_square

        V = tf.eye(3,3) + B*omega_cross + C*omega_cross_square
        Vu = tf.matmul(V,tf.expand_dims(u,1))
        # Vu = tf.expand_dims(u,1)
        T = tf.concat([R, Vu], 1)
        one = np.array([0.,0.,0.,1.]).reshape(1,4)
        T = tf.concat([T, one], 0)
        return T

def mat_to_angle(r):
    t = np.sqrt(r[2, 0] ** 2 + r[2, 2] ** 2)
    angle_z = math.atan2(r[1, 0], r[0, 0])
    angle_y = math.atan2(-1 * r[2, 0], t)
    angle_x = math.atan2(r[2, 1], r[2, 2])
    return np.array([angle_x, angle_y, angle_z])



