from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
# from . import data_augment
from keras_frcnn import data_augment
import threading
import itertools

#计算并的面积
def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union

#计算相交面积
def intersection(ai, bi): #(x1,y1,x2,y2)对应左上，右下两个角点，图片的坐标原点在左上，向右向下都变大
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):  #iou范围（0，1） 计算交并比
	# a and b should be (x1,y1,x2,y2)   （x1,y1）左  （x2,y2）右下

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600): #按比例缩放图片
	if width <= height:
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

'''
训练信息类
图片信息
图片宽度
图片高度（重新计算bboxes要用）
规整化后图片宽度
规整化后图片高度
计算特征图大小函数'''
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
	#返回anchor是否包含类，和回归梯度
#c是config信息，  width height 是原图像的不是经过处理的标准输入的尺寸P*Q，resized 是标准输入M*N
	#  img_length_calc用来计算特征图大小的函数

	downscale = float(C.rpn_stride)   #16
	anchor_sizes = C.anchor_box_scales  #[128, 256, 512]
	anchor_ratios = C.anchor_box_ratios  #[1:1,1:2,2:1]
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	 #3*3

	# calculate the output map size based on the network architecture
#从图片尺寸计算出经过网络之后特征图的尺寸
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)  #

	n_anchratios = len(anchor_ratios)  #3
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))  #[outh,outw,9]
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))  #[outh,outw,9] 有效框
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))  #[outh,outw,9*4]  回归坐标

	num_bboxes = len(img_data['bboxes'])  #统计一张图上 bbox的个数，可以是多个
    #每个bbox对应9个anchor，但是只存储一个bbox对应最好的那个anchor的数据
	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)  #[ numbbox个0 ]  bbox对应的anchor数量
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)  #[numbbox,4 ]填充-1  bbox最好的anchor
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)  #[ numbbox个0    ] bbox对应的最好iou值
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)   #[numbbox,4 ]填充0   对应某个bbox的最好的anchor坐标
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)  #[numbbox,4 ]填充0 最好的 修正anchor参数

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))   #[numbbox,4 ]填充0  初始化用于获取bbox 的gt的坐标变量
	for bbox_num, bbox in enumerate(img_data['bboxes']):   #一个放编号，一个放bbox值
		# get the GT box coordinates, and resize to account for image resizing
        #存4个坐标到gta中  缩放匹配到resize以后的图像bbox的大小
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	# rpn ground truth

	for anchor_size_idx in range(len(anchor_sizes)):  #遍历每个anchor的尺寸【128，256，512】
		for anchor_ratio_idx in range(n_anchratios):   #遍历每个anchor的比例【1：1.1：2，2：1】
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
			
			for ix in range(output_width):  #在网络输出featuremap上的每一个点，执行操作
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2  #x的左下点   对应原图，因为成了缩放因子16
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	#x的右上点 同上
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width:  #resizedwidth 指的是原图M和N，不是网络处理后的图
					continue
					
				for jy in range(output_height):
                    #在网络输出featuremap上的每一个点，执行下面操作
					# y-coordinates of the current anchor box 对y也执行上面的检查
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue
				# '''output_width，output_height：特征图的宽度与高度
                # downscale：将特征图坐标映射到原图的比例
                # if语句是将超出图片的框删去'''
					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'   #用来指示anchor是否包含目标

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0  #用来指示对当前坐标和anchor最佳的iou值

					for bbox_num in range(num_bboxes):  #遍历图片中，所有的bbox框（这个是做数据是标定的）
						
						# get IOU of the current GT box and the current anchor box 计算gta与当前anchor的交集
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
                        #如果交集大于best_iou_for_bbox[bbox_num]或者大于我们设定的阈值，
                        # 就会去计算gta和anchor的中心点坐标，再通过中心点坐标和bbox坐标，计算出x,y,w,h四个值的梯度值
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0
                            #下面是一个计算线性 bounding box regression原理
							tx = (cx - cxa) / (x2_anc - x1_anc)   #anchor和gt的x坐标中心差/anchor的宽
							ty = (cy - cya) / (y2_anc - y1_anc)   #anchor和gt的y坐标中心差/anchor的高
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						if img_data['bboxes'][bbox_num]['class'] != 'bg': #前提是这个bbox的class不是'bg'背景

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:  #如果iou值大于最好的box的iou  bestiouforbbox初始化为0
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]  #将featuremap中像素点坐标和anchor 比例 大小写道bestanchorforbbox
								best_iou_for_bbox[bbox_num] = curr_iou  #将iou值写到besetiouforbbox
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]  #anchor坐标写入
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]  #将修正偏移写入

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:  #如果iou值大于rpnmaxoverlap
								bbox_type = 'pos'  #bbox就是正的
								num_anchors_for_bbox[bbox_num] += 1  #对应的bbox的对应anchor要+1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc: #如果当前的iou 大于loc最好的iou值
									best_iou_for_loc = curr_iou  #当前iou赋值给bestiouforloc
									best_regr = (tx, ty, tw, th)  #回归偏移

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos': #如果bbox不是正例 设成中立
									bbox_type = 'neutral'
                    #依然对feature map每个坐标点操作，如果bbox类型是neg 那么这一点是是有效的预选框但是不包含物体
                    #如果bbox是中性的，它既不包含物体，也不是有效框
                    #如果bbox是pos，他是有效的，也包含物体
					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1  #预选框是否有效
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0  #存储是否包含物体
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr   #梯度回归

	# we ensure that every bbox has at least one positive RPN region
	#有一个bbox没有pos的预选宽和其对应，这找一个与它交并比最高的anchor的设置为pos
	for idx in range(num_anchors_for_bbox.shape[0]):  #遍历每一个图片中bbox数量
		if num_anchors_for_bbox[idx] == 0:   #如果对应bbox对应的anchor数为0
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1: #
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
		# '''从可用的预选框中选择num_regions
		# 如果pos的个数大于num_regions / 2，则将多下来的地方置为不可用。如果小于pos不做处理
		# 接下来将pos与neg总是超过num_regions个的neg预选框置为不可用'''
	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

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
#定义获取gt数据信息
def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			np.random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation

				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				(width, height) = (img_data_aug['width'], img_data_aug['height'])
				(rows, cols, _) = x_img.shape

				assert cols == width
				assert rows == height

				# get image dimensions for resizing
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)   #调整图片大小

				# resize the image so that smalles side is length = 600px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

				try:
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)  #计算rpn网络输出
				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (2, 0, 1))
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				if backend == 'tf':
					x_img = np.transpose(x_img, (0, 2, 3, 1))
					y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
					y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
