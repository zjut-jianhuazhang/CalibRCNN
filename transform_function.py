import tensorflow as tf
import config_res as config
import cv2
import numpy as np
import copy


def points2d_to_rgbimg(points_2d, img):

    img_ht = img.shape[0]
    img_wdt = img.shape[1]
    Z = points_2d[2, :]
    x = (points_2d[0, :] / Z).T
    y = (points_2d[1, :] / Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)
    colors_map = np.loadtxt('color_jet.txt', dtype=np.float32)
    colors_map = colors_map * 255

    depth_rgbimg = copy.deepcopy(img)

    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if z_idx > 0:

            idx = np.clip(z_idx, 0.0, 64 - 1)  # 64个颜色梯度
            idx = int(idx);
            x_idx = int(x_idx);
            y_idx = int(y_idx)

            color = tuple([int(c) for c in colors_map[idx, :]])
            cv2.circle(depth_rgbimg, (x_idx, y_idx), 2, color, -1)
    return depth_rgbimg


def rgbd_transform(depimg, trans, k,img):
    # 图片大小
    img_ht = img.shape[0]
    img_wdt = img.shape[1]
    x_idx, y_idx = np.nonzero(depimg)

    points_2d = np.zeros((1,3), dtype = np.float32)

    for x, y in zip(x_idx,y_idx):
        z = depimg[x,y]
        point_2d = np.expand_dims(np.array([y*z,x*z,z]), 0)
        points_2d = np.vstack((points_2d,point_2d))

    points_2d = points_2d[1:] # N*3
    pos = np.matmul(np.linalg.inv(k),points_2d.T)
    ones_cel = np.ones(shape=(1,points_2d.shape[0]))
    pos1 = np.vstack((pos, ones_cel)) # 4*N
    points_in_cam_axis = np.matmul(config.camera_params['cam_transform_02'], pos1) # 4*N
    transformed_points = np.matmul(trans,points_in_cam_axis) # 4*N
    points_2d_pre = np.matmul(k, np.matmul(config.camera_params['cam_transform_02_inv'], transformed_points)[:-1, :])
    Z = points_2d_pre[2, :]
    x = (points_2d_pre[0, :] / Z).T
    y = (points_2d_pre[1, :] / Z).T

    x = np.clip(x, 0.0, img_wdt - 1)
    y = np.clip(y, 0.0, img_ht - 1)
    colors_map = np.loadtxt('color_jet.txt', dtype=np.float32)
    colors_map = colors_map * 255
    reprojected_img = np.zeros_like(img)
    color_img = copy.deepcopy(img)

    for x_idx, y_idx, z_idx in zip(x, y, Z):
        if z_idx > 0:
            reprojected_img[int(y_idx), int(x_idx)] = z_idx
            idx = np.clip(z_idx, 0.0, 64 - 1)
            idx = int(idx);
            x_idx = int(x_idx);
            y_idx = int(y_idx)
            # color_img[int(y_idx), int(x_idx)] = colors_map[idx,:]

            color = tuple([int(c) for c in colors_map[idx, :]])
            cv2.circle(color_img, (x_idx, y_idx), 2, color, -1)
    return color_img

