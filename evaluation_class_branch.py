# torchvision:
# /home/enterprise.internal.city.ac.uk/sbrn151/.local/lib/python3.5/site-packages/torchvision/models/detection/__init__.py
import os
import pickle
import re
import sys
import time
from collections import OrderedDict
import config_ssm as config
import cv2
import datasets.dataset_classification as dataset_classification
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

	# get the thresholds
	box_detections, test_data_dir, rpn_nms_th, box_nms_thresh_classifier \
		= config.box_detections, config.test_data_dir, config.rpn_nms_th, config.box_nms_thresh_classifier

	if model_name is None:
		model_name = "SingleShotModel"
	if config.model_name is not None and config.model_name != model_name:
		print("Using model name from the config.")
		model_name = config.model_name

	# classification dataset interface
	# dataset+dataloader
	dataset_class_pars = {'stage': 'eval', 'data': test_data_dir, 'img_size': (512,512)}
	datapoint_class = dataset_classification.COVID_CT_DATA(**dataset_class_pars)
	dataloader_class_pars = {'shuffle': False, 'batch_size': 1}
	dataloader_class_eval = data.DataLoader(datapoint_class, **dataloader_class_pars)
	# load the weights and create the model
	sizes = ckpt['anchor_generator'].sizes
	aspect_ratios = ckpt['anchor_generator'].aspect_ratios
	anchor_generator = AnchorGenerator(sizes, aspect_ratios)
	print("Anchors: ", anchor_generator)
	# create modules
	box_head_input_size = 256 * 7 * 7
	box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
	box_head = TwoMLPHead(in_channels=box_head_input_size, representation_size=128)
	box_predictor = FastRCNNPredictor(in_channels=128, num_classes=2)
	# Masks are not used for classification
	mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
	mask_rcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
	mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=2)
	#
	classifier_covid = S2Predictor(in_channels=5, batch_size=box_detections, representation_size=512, out_channels=3)
	# keyword arguments
	# box_score_thresh_classifier == -0.01
	ssm_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'detections_per_img': box_detections,
				'box_nms_thresh_classifier': box_nms_thresh_classifier, 'box_score_thresh_classifier': -0.01, 'rpn_nms_thresh': rpn_nms_th,
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
	# confusion matrix
	cmatrix, c_sens, overall_acc, f1 = step(ssm, dataloader_class_eval, device)
	print("Done evaluation for {}".format(model_name))
	end=time.time()
	total_time = end-start
	print("Evaluation time {:.2f} seconds".format(total_time))


# returns confusion matrix, precision and recall derived from it
def main_step(model, dl, device):
	confusion_matrix = torch.zeros(3, 3, dtype=torch.int32).to(device)

	for v, b in enumerate(dl):
		X, y = b
		if device == torch.device('cuda'):
			X, y = X.to(device), y.to(device)
		image = [X.squeeze_(0)]  # remove the batch dimension
		_, pred_scores, _ = model(image)
		# predicted class scores
		confusion_matrix[torch.nonzero(y.squeeze_(0)>0).item(), pred_scores[0]['final_scores'].argmax().item()] += 1
	print("------------------------------------------")
	print("Confusion Matrix for 3-class problem, a total of {0:d} images:".format(len(dl)))
	print("0: Control, 1: Normal Pneumonia, 2: COVID")
	print(confusion_matrix)
	print("------------------------------------------")
	# confusion matrix
	cm = confusion_matrix.float()
	cm[0, :].div_(cm[0, :].sum())
	cm[1, :].div_(cm[1, :].sum())
	cm[2, :].div_(cm[2, :].sum())
	print("------------------------------------------")
	print("Class Sensitivity:")
	print(cm)
	print("------------------------------------------")
	print('Overall accuracy:')
	oa = confusion_matrix.diag().float().sum().div(confusion_matrix.sum())
	print(oa)
	cm_spec = confusion_matrix.float()
	cm_spec[:, 0].div_(cm_spec[:, 0].sum())
	cm_spec[:, 1].div_(cm_spec[:, 1].sum())
	cm_spec[:, 2].div_(cm_spec[:, 2].sum())
	# Class weights: 0, 1, 2
	cw = torch.tensor([0.45, 0.35, 0.2], dtype=torch.float).to(device)
	print("------------------------------------------")
	print('F1 score:')
	f1_score = 2 * cm.diag().mul(cm_spec.diag()).div(cm.diag() + cm_spec.diag()).dot(cw).item()
	print(f1_score)
	# Confusion matrix, class sensitivity, overall accuracy and F1 score
	return confusion_matrix, cm, oa, f1_score


if __name__ == "__main__":
	config_class = config.get_config_pars_ssm("test_classification")
	main(config_class, main_step)
