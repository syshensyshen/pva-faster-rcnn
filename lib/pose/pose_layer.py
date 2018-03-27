# --------------------------------------------------------
# Face Pose
# Copyright (c) 2017 Facevisa
# Licensed under The MIT License [see LICENSE for details]
# Written by syshen
# --------------------------------------------------------

"""This layer used during training to train a pose regression network.

pose_layer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import copy
import yaml
import random

class PoseLayer(caffe.Layer):

    def generate_bbox(self, rect, ratios):
	x = np.float32(rect[0])
	y = np.float32(rect[1])
	w = np.float32(rect[2])
	h = np.float32(rect[3])
	center_x = x + w/2
	center_y = y + h/2
	index = 0
	scales = np.array(([0.85, 1, 1.2, 1.5]))
	#scales = np.array(([1.5]))
	if ratios.shape[0] == 2:
	   new_rect = np.zeros([4, 4], np.int32)
	   for scale in  scales:
	        w_s = w * scale
		h_s = h * scale
		new_rect_elem = np.zeros([4], np.int32)
		new_rect_elem[0] = np.int32(center_x - w_s / 2 * ( 1 + ratios[0]))
		new_rect_elem[1] = np.int32(center_y - h_s / 2 * (1 + ratios[1]))
		new_rect_elem[2] = np.int32(w_s)
		new_rect_elem[3] = np.int32(h_s)
		new_rect[index, :] = new_rect_elem
		index = index + 1
	else:
	   new_rect = np.zeros([ratios.shape[0]*4, 4], np.int32)
	   for ratio in ratios:
		for scale in  scales:
			w_s = w * scale
			h_s = h * scale
			new_rect_elem = np.zeros([4], np.int32)
			new_rect_elem[0] = np.int32(center_x - w_s / 2 * ( 1 + ratio[0]))
			new_rect_elem[1] = np.int32(center_y - h_s / 2 * (1 + ratio[1]))
			new_rect_elem[2] = np.int32(w_s)
			new_rect_elem[3] = np.int32(h_s)
			new_rect[index, :] = new_rect_elem
			index = index + 1
	return new_rect


    def generate_all_samples(self, gt_boxes, poses):
	bbox_num = gt_boxes.shape[0]
	all_bboxes = np.zeros([bbox_num, 4], np.float32)
	all_poses = np.zeros([bbox_num, 3], np.float32)
	scale_elem = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        # samples join in regression
	gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
	gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
	all_bboxes[0:bbox_num, :] = gt_boxes
	all_poses[0:bbox_num, :] = poses
	ratios = np.random.random(5)/10 + 0.05
	#ratios[0] = 0.01
	#print ratios
	index = 0
	while index <= bbox_num * 8:
		bboxes = self.generate_bbox(gt_boxes[index%bbox_num], np.array(ratios[index%5]) * scale_elem)
		#bboxes = generate_bbox(gt_boxes[index%bbox_num], np.array(ratios[0]) * scale_elem)
		pose = np.zeros([16, 3], np.float32)        
		all_bboxes = np.row_stack((all_bboxes, bboxes))
		pose[0:16, :] = poses[index%bbox_num]
		all_poses = np.row_stack((all_poses, pose))
		index = index + 1
	#all_bboxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
	#all_bboxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
	#print all_bboxes
	#raw_input()
	return all_bboxes, all_poses
	
    def setup(self, bottom, top):
	# parse the layer parameter string, which must be valid YAML
	layer_params = yaml.load(self.param_str)
	self._num_bboxes = layer_params['num_bboxes']
	if len(bottom) != 3:
		raise Exception("Need to define two bottoms.")
	# data layers have no bottoms
	if len(top) != 4:
		raise Exception("top num mismatch.")
	# sample label and bounding box
	top[0].reshape(self._num_bboxes, 5)
	# sample label and pose
        top[1].reshape(self._num_bboxes, 1)
	top[2].reshape(self._num_bboxes, 1)
	top[3].reshape(self._num_bboxes, 1)

    def reshape(self, bottom, top):
        pass
	
    def forward(self, bottom, top):
	gt_boxes = bottom[0].data
        pose_label = bottom[1].data
        im_info = bottom[2].data[0, :]
	height = np.int32(im_info[0])
	width = np.int32(im_info[1])

        bboxes, poses = self.generate_all_samples(copy.copy(gt_boxes[:, 0:4]), pose_label)
        # limit bounding in image shape
        #height, width = np.int32(im_info.shape[:2])
        #bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
        #bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
        for bbox in bboxes:
            if bbox[0] <= 0.1:
                bbox[0] = 4
            if bbox[1] <= 0.1:
                bbox[1] = 4
            if bbox[0] + bbox[2] >= width - 0.1:
                bbox[2] = width - bbox[2] - 4
            if bbox[1] + bbox[3] >= height - 0.1:
                bbox[3] = height - bbox[3] - 4
        # select bbox random
        random_ = range(0, bboxes.shape[0])
        #print bboxes.shape[0]
        select_bbox = random.sample(random_, 96)
        bboxes = bboxes[select_bbox, :]
        poses = poses[select_bbox, :]
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        bboxes[96 - gt_boxes.shape[0]:96, :] = gt_boxes[:, 0:4]
        poses[96 - gt_boxes.shape[0]:96, :] = pose_label
        # convert angle to radian
        poses = poses*180/3.14
	#print gt_boxes
	#print height, width
	#print bboxes
	#print poses
	#raw_input()
        batch_inds = np.zeros((bboxes.shape[0], 1), dtype=np.float32)
        blob_0 = np.hstack((batch_inds, bboxes.astype(np.float32, copy=False)))
        top[0].reshape(*(blob_0.shape))
        top[0].data[...] = blob_0
        #blob_1 = np.hstack((batch_inds, poses.astype(np.float32, copy=False)))
        #top[0].reshape(*(blob_1.shape))
        #top[0].data[...] = blob_1
        #top[1].reshape(*(blob_1.shape))
        #top[1].data[...] = poses # old regression
	top[1].data[:, 0] = poses[:, 0]
	top[2].data[:, 0] = poses[:, 1]
	top[3].data[:, 0] = poses[:, 2]
	#print poses[:,0].shape
	#raw_input()

    def backward(self, top, propagate_down, bottom):
	"""This layer does not propagate gradients."""
        pass
