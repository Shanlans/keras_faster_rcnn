from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools


def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1]) # GT 面积， （x2-x1）*(y2-y1)
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1]) # BBOX 面积， (X2-X1)*(Y2-Y1)
	area_union = area_a + area_b - area_intersection # 算面积并集
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0]) # GT 与 bbox 之间最大的X1
	y = max(ai[1], bi[1]) # GT 与 bbox 之间最大的Y1
	w = min(ai[2], bi[2]) - x # GT 与bbox 之间最小的X2 - x 为交叠区域的 col
	h = min(ai[3], bi[3]) - y # GT 与bbox 之间最小的Y2 - x 为交叠区域的 row
	if w < 0 or h < 0:  # 如果w或者h小于0 说明没有交叠，返回0
		return 0
	return w*h # 否则返回面积


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]: # a 是 ground truth， b 是bounding box， 不可以x1 >= x2 y1>=y2
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	if width <= height:   #论文中说，要找最小边，并将最小边resize 到 600，即最长边也按比例resize
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):

	downscale = float(C.rpn_stride) # C.rpn_stride 表示的是 使用 基础网络downscale 的尺寸是rpn_stride, 16倍 downscale
	anchor_sizes = C.anchor_box_scales #anchor box 的尺寸 ，论文中是 [128,256,512]
	anchor_ratios = C.anchor_box_ratios #anchor box 的 ratio， 论文中是[1:1,1:2,2:1]
	num_anchors = len(anchor_sizes) * len(anchor_ratios) # 最后应该是 9 个 anchor box

	# calculate the output map size based on the network architecture
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height) #计算特征提取基础网络最后输出的大小

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors)) # 初始化 anchor 容器？，特征提取层会输出output_height*output_width 的feature map 以及每个点对应的9个anchors
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors)) # 初始化 anchor 是否起效的容器，与anchor容器大小一致
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4)) # 初始化 每个region porposal 坐标容器，anchor 数目*4

	num_bboxes = len(img_data['bboxes']) # 每个图的bbox个数

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int) # 初始化 统计量 每个bbox对应了多少个最好的anchor（IOU 最大或者 IOU >70% 时，对应bbox的数值+1）
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) #初始化 每个bbox 对应的当前IOU最好的anchor box 的 中心坐标以及 anchor ratio和size
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32) # 初始化每个bbox与anchor 最好的 iou 初始值 为 0。对应论文中，对bbox回归需要二级判定，是否为70% iou或者是最好的iou
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int) # 初始化当前bbox 对应的 最好的 anchor 在原图中的 x1,x2,y1,y2的位置
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32) # 初始化当前bbox 对应的 anchor位置偏移量，即中心点坐标以及长宽

	# get the GT box coordinates, and resize to account for image resizing
	# convert the label bbox to resized image, just change the ratio according to resize/original_size
	gta = np.zeros((num_bboxes, 4)) # 初始化 ground truth 中的bbox 的坐标值，因为要resize image ，所以bbox 也要resize
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth
	# iterate anchor size and ratio, get all possiable RPNs
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios): ## 前两层循环选出单一 anchor尺寸
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			
			# TODO: Important part, we got final feature map output_w and output_h
			# then we reflect back every anchor positions in the original image
			for ix in range(output_width):
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2 # ix 为最后 feature 输出的每个点的横坐标，+0.5是要将坐标移至该像素点中心，然后根据downscale也就是累计stride来映射回原始位置在哪
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2 # X1 ，X2 值
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width: # 如果 某一个 真实anchor 超过了 原图边界，则跳过
					continue
					
				for jy in range(output_height): ## 这两层循环选出 feature map 上一个点 并得出这个点所对应的 anchor

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes): # 这层循环，遍历所有的 GT 中的bbox 看是否在这个点上有对应的 bbox，并且得出这个bbox和当前所选择尺寸的anchor 的关系
						
						# get IOU of the current GT box and the current anchor box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap: # 计算当前iou是不是bbox与这个尺寸anchor的最好iou，或者是否大于70%
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0 # GT x 中心位置
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0 # GT y 中心位置
							cxa = (x1_anc + x2_anc)/2.0 # anchor x 中心位置
							cya = (y1_anc + y2_anc)/2.0 # anchor y 中心位置

							tx = (cx - cxa) / (x2_anc - x1_anc) #论文 3.1.2 节 tx = (x-xa)/wa     # 这里面cx、cy、cxa、cya都是中心坐标
							ty = (cy - cya) / (y2_anc - y1_anc) # ty = (y-ya)/ha
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]: # 如果当前的IOU 超过属于这个bbox的最好IOU 则更新以下数值
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx] # 当前anchor box 的 中心坐标（即feature map对应位置）以及 anchor ratio和size
								best_iou_for_bbox[bbox_num] = curr_iou # 更新这个bbox最好的IOU
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc] #当前anchor的四个位置映射到原图的真实位置
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th] # 当前bbox 对应的 anchor位置偏移量，即中心点坐标以及长宽便宜系数

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1 # 只要当前bbox 的IOU 大于 70% 就为这个bbox的anchor 数量 +1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc: # 记录 这个最好的IOU 对应的 anchor
									best_iou_for_loc = curr_iou # 这个anchor最好 更新最好 IOU
									best_regr = (tx, ty, tw, th) # 如果这个anchor最好，最后 位置回归的时候就用这个anchor相对于bbox的位置偏移量

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
					# box type has neg,pos, neutral
					# neg: the box is backgronud
					# pos: the box has RPN
					# neutral: normal box with a RPN

					#循环完所有的gt后，会对bbox_type状态进行改变
					#1. 无论哪个bbox与当前指定的anchor都不符合要求，bbox_type状态不发生改变，就说明在这个feature map 位置上，box valid 设定为起效，但是overlap设为不起效；
					#2. 无论哪个bbox与当前指定的anchor都符合一项要求（IOU 在 30~70之间），bbox_type状态发生改变 变为 neutral，box valid 设定为不起效，overlap设为不起效；
					#3. 无论哪个bbox与当前指定的anchor符合要求，都说明在这个feature map 位置上，box valid 设定为起效，overlap 也起效

					# 这时要注意，因为是先遍历bbox 再变历所有的anchor，大的bbox很容易就会让当前位置所有尺寸的anchor ‘valid’
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 # anchor_ratio_idx + n_anchratios * anchor_size_idx计算的是第几个anchor
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr # 收集每个位置最好的anchor

	# we ensure that every bbox has at least one positive RPN region
	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		# 如果positive location 大于 128，则随机关闭一些 positive loc （如果是256个，则随机关闭128个）
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		# 如果negive location 与 positive的location 的和大于128，则关闭那些比 positive 个数多的（如果positive 个数等于128 则关闭多余128个，如果postive个数小于128，则关闭到跟postive个数一致的）
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1) # [:,18,...] 对应
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
	# [:,72,...] 这里[:,0:36,...]为了后续rpn regression loss计算中判定是不是当前bbox计算loss，当然[:,36:,...]就是所有的reg location

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation

				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				(width, height) = (img_data_aug['width'], img_data_aug['height']) # 做完 argument 的图像尺寸，可能有翻转
				(rows, cols, _) = x_img.shape # 应与 image_data_aug一致

				assert cols == width
				assert rows == height

				# get image dimensions for resizing
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size) # 最小边 resize成 600，最大边等比例resize

				# resize the image so that smalles side is length = 600px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

				try:
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (2, 0, 1)) # RGB -> BRG？？？？
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling ## ？？？

				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1)) # 神经病 BRG -> RGB
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1)) # cls,y,x ->y,x,cls
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) # reg,y,x -> y,x,reg

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
