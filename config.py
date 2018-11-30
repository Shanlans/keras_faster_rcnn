anchor_box_ratios ： {list} <class 'list'>: [[1, 1], [1, 2], [2, 1]]
anchor_box_scales ： {list} <class 'list'>: [128, 256, 512]
balanced_classes ： {bool} False
base_net_weights ： {str} './model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
class_mapping ： {dict} {'Misc': 5, 'Cyclist': 3, 'bg': 9, 'DontCare': 4, 'Truck': 1, 'Tram': 7, 'Car': 2, 'Van': 6, 'Pedestrian': 0, 'Person_sitting': 8}
classifier_max_overlap ： {float} 0.5
classifier_min_overlap ： {float} 0.1
classifier_regr_std ： {list} <class 'list'>: [8.0, 8.0, 4.0, 4.0]
config_save_file ： {str} 'config.pickle'
data_dir ： {str} '.data/'
im_size ： {int} 600
img_channel_mean ： {list} <class 'list'>: [103.939, 116.779, 123.68]
img_scaling_factor ： {float} 1.0
kitti_simple_label_file ： {str} 'kitti_simple_label.txt'
model_path ： {str} './model/kitti_frcnn_last.hdf5'
network ： {str} 'resnet50'
num_epochs ： {int} 3000
num_rois ： {int} 32
rot_90 ： {bool} True
rpn_max_overlap ： {float} 0.7
rpn_min_overlap ： {float} 0.3
rpn_stride ： {int} 16
simple_label_file ： {str} 'kitti_simple_label.txt'
std_scaling ： {float} 4.0
use_horizontal_flips ： {bool} True
use_vertical_flips ： {bool} True
verbose ： {bool} True