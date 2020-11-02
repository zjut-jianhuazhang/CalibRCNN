import numpy as np

# Important paths

# resnet_params_path: path containing .pkl file of RESNET18_BN weights
# dataset_path_full: path to parser file
# checkpoint_path: dir where checkpoints are saved and loaded from
# training_imgs_path: dir to save training progress frames (spatial transformer outputs during training)
# validation_imgs_path: dir to save validation progress frames (spatial transformer outputs during validation)
paths = dict(
        dataset_path_full = "../dataset/parsed_set.txt",
        checkpoint_path = "Checkpoint_simple_transformer",
        training_imgs = "../training_imgs/",
        testing_imgs = "../testing_imgs/",
        validation_imgs = "../validation_imgs/",
        transformed_depth = "../dataset/transformed_depth/",

        test_data = "../dataset/test_data_list1.txt",
        train_data = "../dataset/train_data_list1.txt",
        match_point = "../dataset/match_points/",
        cam_img = "../dataset/cam/",
        # cam_pose = "../dataset/data_odometry_poses/dataset/poses/00.txt",
        resnet_params_path = "parameters.json"
)


# Depth Map parameters

# IMG_HT: input image height
# IMG_WDT: input image width
depth_img_params = dict(
        IMG_HT = 376, # 376
        IMG_WDT = 1241 # 1241
)






# Network and Training Parameters

# batch_size: batch_size taken during training
# total_frames: total instances (check parsed_set.txt for total number of lines)
# total_frames_train: total training instances
# total_frames_validation: total validation instances
# partition_limit: partition size of the total dataset loaded into memory during training
# epochs: total number of epochs
# learning_rate
# beta1: momentum term for Adam Optimizer
# load_epoch: Load checkpoint no. 0 at the start of training (can be changed to resume training)
net_params = dict(
        batch_size = 4,
        load_batch = 1,
        epochs = 400,
        learning_rate = 1e-4, # 0.0002 5e-5
        beta1 = 0.9,
        load_epoch = 0,
        total = 20000, # 4036
        total_train= 20000, # 3228
        total_validation = 24,
        total_test = 600 # 792

	)
