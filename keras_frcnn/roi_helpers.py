import numpy as np
import pdb
import math
from . import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []  # for debugging only

    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                           [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    '''
    X do the transformation by T
    :param X: A[:,:,:,curr_layer]：shape[4,W,H] feature map中，当前anchor尺寸的所有 anchor中心点和长宽
    :param T: regr: shape[4,W,H] feature map中，当前回归预测的，anchor相对于可以预测到物体的偏移量 tx ty tw th
    :return:
    '''
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2. # 计算中心点 cx
        cy = y + h / 2. # 计算中心点 cy
        cx1 = tx * w + cx # 叠加偏移量，计算真正中心点 cx1
        cy1 = ty * h + cy # 叠加偏移量，计算真正中心点 cy1

        w1 = np.exp(tw.astype(np.float64)) * w #叠加 anchor 偏移量，计算真正的宽
        h1 = np.exp(th.astype(np.float64)) * h #叠加 anchor 偏移量，计算真正的长
        x1 = cx1 - w1 / 2. #通过长和宽以及中心点，计算出物体的左上角定点
        y1 = cy1 - h1 / 2. #

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    # I changed this method with boxes already contains probabilities, so don't need prob send in this method
    # TODO: Caution!!! now the boxes actually is [x1, y1, x2, y2, prob] format!!!! with prob built in
    if len(boxes) == 0:
        return []
    # normalize to np.array
    boxes = np.array(boxes)
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2) #又判断了遍 x1<x2
    np.testing.assert_array_less(y1, y2) #又判断了遍 y1<y2

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    area = (x2 - x1) * (y2 - y1)
    # sorted by boxes last element which is prob
    indexes = np.argsort([i[-1] for i in boxes]) # 将所有bounding box的概率值从小到大排列后得到顺序index

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last] # 从最大的开始取
        pick.append(i) # 记录最大的bbox的序号

        # find the intersection
        # 极大值抑制 https://blog.csdn.net/shuzfan/article/details/52711706
        # 将概率最大的bbox的x1,y1,x2,y2与所有其他概率值宇轩框的x1,y1,x2,y2进行比较，先找出大于x1,y1小于x2,y2的预选框
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        # intersection 面积计算
        area_int = ww_int * hh_int
        # find the union
        area_union = area[i] + area[indexes[:last]] - area_int # 计算所有的bbox与最大的bbox之间的并集

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6) # IOU

        # delete all indexes from the index list that have
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))) # 将IOU 大于一定值的bbox与prob最大的bbox delete掉

        if len(pick) >= max_boxes: #当选取得bbox个数超过 300个时，退出
            break
    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick] # 返回最大300个 （x1,x2,y1,y2,prob）
    return boxes


def rpn_to_roi(rpn_layer, regr_layer, cfg, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """

    :param rpn_layer: np.array. [BATCH,W,H,9] 返回的是每个anchor box cls 类别，是否是前景，还是背景
    :param regr_layer: np.array. 返回的是每个前景的regrer 在原图尺寸（当然是resize后的）上的偏移尺寸
    :param cfg:
    :param dim_ordering:
    :param use_regr: Boolean.
    :param max_boxes: Integer. Default 300. Test time using maximum proposal area number. In the paper section 4.1
    :param overlap_thresh:
    :return:
    """
    # 本模块将rpn输出的region proposals （即选定的anchor）对真实bbox位置偏移量。映射到 feature map上的 ROI。 从大映射到小。

    regr_layer = regr_layer / cfg.std_scaling # 为什么要有这个值？

    anchor_sizes = cfg.anchor_box_scales
    anchor_ratios = cfg.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'th':
        (rows, cols) = rpn_layer.shape[2:]

    elif dim_ordering == 'tf':
        (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    if dim_ordering == 'tf':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3])) # 4个coordinates, feature map row, feature map col, 9个 anchor
    elif dim_ordering == 'th':
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:   # 选定一个anchor尺寸 作为reference

            anchor_x = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride
            if dim_ordering == 'th':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  #遍历每个regr_layer中36个anchor相对于真实物体BBOX的4个偏移量， [W,H,4]
                regr = np.transpose(regr, (2, 0, 1)) #[W,H,4] -> [4,W,H]

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows)) # 生成X,Y每个坐标值

            A[0, :, :, curr_layer] = X - anchor_x / 2 # 以每个feature map的点，作为anchor的重点，计算出该anchor对应的左上角顶点以及长宽 （curr_layer 从0 到 9 ）
            A[1, :, :, curr_layer] = Y - anchor_y / 2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                # A[:,:,:,curr_layer]：shape[4,W,H] feature map中，当前anchor尺寸的所有 anchor中心点和长宽
                # regr: shape[4,W,H] feature map中，当前回归预测的，anchor相对于可以预测到物体的偏移量 tx ty tw th
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr) # 返回 [x1, y1, w1, h1]，左上角点以及物体长宽

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer]) #预测的bbox经过变换后的W 不能小于1
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer]) #预测的bbox经过变换后的H 不能小于1
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer] #预测的bbox的宽变为右下角的点的x2 = x1 + W
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer] #预测的bbox的长变为右下角的点的y2 = y1 + H

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer]) #预测的bbox经过变换后的x1 不能小于0
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer]) #预测的bbox经过变换后的y1 不能小于0
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer]) #预测的bbox经过变换后的x2 不能大于长（cols-1）
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer]) #预测的bbox经过变换后的y2 不能大于宽（rows-1）

            curr_layer += 1 # 下一个anchor

    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0)) #所有 W*H*9 个bounding box 的 4个坐标值 [W*H*9,4]
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))  #所有 W*H*9 个bounding box 的cls （前景还是后景） [W*H*9]

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

    all_boxes = np.delete(all_boxes, ids, 0) # 删除 (x1,y1),(x2,y2)位置不正常的点
    all_probs = np.delete(all_probs, ids, 0) # 删除 (x1,y1),(x2,y2)位置不正常的概率值

    # I guess boxes and prob are all 2d array, I will concat them
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs]))) # [W*H*9,5] 5的顺序为 X1,Y1,X2,Y2,PROB
    result = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes)
    # omit the last column which is prob
    result = result[:, 0: -1]
    return result
