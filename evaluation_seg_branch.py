import os
import pickle
import re
import sys
import time
from collections import OrderedDict
import config_ssm as config
import cv2
import datasets.dataset_segmentation as dataset
import matplotlib.pyplot as plt
import models.ssm
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import utils
from PIL import Image as PILImage
from models.ssm.faster_rcnn_ssm import FastRCNNPredictor, TwoMLPHead, S2Predictor
from models.ssm.rpn import AnchorGenerator
from models.ssm.single_shot_model import *


def main(config, step):
	start = time.time()
	devices = ['cpu', 'cuda']
	assert config.device in devices
	if config.device == 'cuda' and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	#
	model_name = None
	ckpt = torch.load(config.ckpt, map_location=device)
	if 'model_name' in ckpt.keys():
		model_name = ckpt['model_name']

	device = torch.device('cpu')
	if torch.cuda.is_available():
		device = torch.device('cuda')
	# get the thresholds
	box_detections, confidence_threshold, mask_threshold, data_dir, imgs_dir, gt_dir, rpn_nms_th, roi_nms_th \
		= config.box_detections, config.confidence_th, config.mask_logits_th, config.test_data_dir, config.imgs_dir, \
		  config.gt_dir, config.rpn_nms_th, config.roi_nms_th

	if model_name is None:
		model_name = "SingleShotModel"
	if config.model_name is not None and config.model_name != model_name:
		print("Using model name from the config.")
		model_name = config.model_name

	# dataset interface
	# only merged mask implemented and batch_size = 1
	dataset_seg_eval_pars = {'stage': 'eval', 'gt': os.path.join(data_dir, gt_dir),
							   'data': os.path.join(data_dir, imgs_dir), 'mask_type': 'merge', 'ignore_small':True}
	datapoint_seg_eval = dataset.CovidCTData(**dataset_seg_eval_pars)
	dataloader_seg_eval_pars = {'shuffle': False, 'batch_size': 1}
	dataloader_seg_eval = data.DataLoader(datapoint_seg_eval, **dataloader_seg_eval_pars)
	#
	# MASK R-CNN model
	# Alex: these settings could also be added to the config
	ckpt = torch.load(config.ckpt, map_location=device)
	sizes = ckpt['anchor_generator'].sizes
	aspect_ratios = ckpt['anchor_generator'].aspect_ratios
	anchor_generator = AnchorGenerator(sizes, aspect_ratios)
	print("Anchors: ", anchor_generator)
	# create modules
	box_head_input_size = 256 * 7 * 7
	box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
	box_head = TwoMLPHead(in_channels=box_head_input_size, representation_size=128)
	box_predictor = FastRCNNPredictor(in_channels=128, num_classes=2)
	mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
	mask_rcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
	mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=2)
	classifier_covid = S2Predictor(in_channels=5, batch_size=box_detections, representation_size=512, out_channels=3)
	# keyword arguments
	ssm_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'detections_per_img': 128,
				'box_nms_thresh': roi_nms_th, 'box_score_thresh': confidence_threshold, 'rpn_nms_thresh': rpn_nms_th,
				'box_head': box_head,
				'rpn_anchor_generator': anchor_generator, 'mask_roi_pool': mask_roi_pool,
				'mask_predictor': mask_predictor, 'box_predictor': box_predictor, 'mask_head': mask_rcnn_heads,
				's2predictor': classifier_covid}
	print(ssm_args)
	# Instantiate SSM
	ssm = maskrcnn_resnet_fpn(backbone_name='resnet18', pretrained=False, **ssm_args)
	# Load weights
	ssm.load_state_dict(ckpt['model_weights'])
	# Set to the evaluation mode
	print(ssm)
	ssm.eval().to(device)
	# IoU thresholds. By default the model computes AP for each threshold between 0.5 and 0.95 with the step of 0.05
	thresholds = torch.arange(0.5, 1, 0.05).to(device)
	mean_aps_all_th = torch.zeros(thresholds.size()[0]).to(device)
	ap_th = OrderedDict()
	# run the loop for all thresholds
	for t, th in enumerate(thresholds):
		# main method
		ap = step(ssm, th, dataloader_seg_eval, device, mask_threshold)
		mean_aps_all_th[t] = ap
		th_name = 'AP@{0:.2f}'.format(th)
		ap_th[th_name] = ap
	print("Done evaluation for {}".format(model_name))
	print("mAP:{0:.2f}".format(mean_aps_all_th.mean().item()))
	for the, p in ap_th.items():
		print("{0:}:{1:.4f}".format(the, p))
		end=time.time()
		total_time = end-start
		print("Evaluation time {:.2f} seconds".format(total_time))

# MS COCO 2017 criterion
def compute_map(model, iou_th, dl, device, mask_th):
	mean_aps_this_th = torch.zeros(len(dl), dtype=torch.float)
	for v, b in enumerate(dl):
		X, y = b
		if device == torch.device('cuda'):
			X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y[
				'masks'].to(device)
		lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0), 'masks': y['masks'].squeeze_(0)}
		image = [X.squeeze_(0)]  # remove the batch dimension
		_, _,out = model(image)
		# scores + bounding boxes + labels + masks
		scores = out[0]['scores']
		bboxes = out[0]['boxes']
		classes = out[0]['labels']
		predict_mask = out[0]['masks'].squeeze_(1) > mask_th
		if len(scores) > 0 and len(lab['labels']) > 0:
			ap, _, _, _ = utils.compute_ap(lab['boxes'], lab['labels'], lab['masks'], bboxes, classes, scores,
						predict_mask, iou_threshold=iou_th)
			mean_aps_this_th[v] = ap
		elif not len(scores) and not len(lab['labels']):
			mean_aps_this_th[v] = 1
		elif not len(scores) and len(lab['labels']) > 0:
			continue
		elif len(scores) > 0 and not len(y['labels']):
			continue
	return mean_aps_this_th.mean().item()


if __name__ == "__main__":
	config_mean_ap = config.get_config_pars_ssm("test_segmentation")
	main(config_mean_ap, compute_map)
