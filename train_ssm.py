# Single Shot Model For COVID-19 Prediction and Lesion Detection
# Alex Ter-Sarkisov @ City, University of London
# alex.ter-sarkisov@city.ac.uk
#
import os
import pickle
import re
import sys
import time
import config_ssm
import cv2
import datasets.dataset_classification as dataset_classification
import datasets.dataset_segmentation as dataset_segmentation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import models.ssm
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils as utils
import torchvision
import utils
from PIL import Image as PILImage
from matplotlib.patches import Rectangle
from models.ssm.single_shot_model import *
from models.ssm.faster_rcnn_ssm import FastRCNNPredictor, TwoMLPHead, S2Predictor
from models.ssm.rpn import AnchorGenerator
from torch.utils import data
from torchvision import transforms
from torchvision.ops import MultiScaleRoIAlign


# main method
def main(config, main_step):
    torch.manual_seed(time.time())
    start_time = time.time()
    devices = ['cpu', 'cuda']
    assert config.device in devices
    start_epoch, model_name, num_epochs, save_dir, train_classification_data_dir,\
    train_segmentation_data_dir, imgs_dir, gt_dir, device,\
    save_every, lrate, rpn_nms_th, box_detections, checkpoint, box_nms_thresh_classifier = \
    config.start_epoch, config.model_name, config.num_epochs, config.save_dir, config.train_class_data_dir, \
    config.train_seg_data_dir, config.imgs_dir, config.gt_dir, config.device, config.save_every,\
    config.lrate, config.rpn_nms_th, config.box_detections, config.checkpoint, config.box_nms_thresh_classifier

    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    use_pretrained=False
    if checkpoint is not None:
        use_pretrained = True
    if save_dir not in os.listdir('.'):
       os.mkdir(save_dir)
    ##############################################################################################
    # DATASETS+DATALOADERS
    # Alex: could be added in the config file in the future
    # parameters for the classification dataset/dataloaders
    # 512x512 is the recommended image size input
    # Only for batch_size=1
    dataset_class_pars = {'stage': 'train', 'data': train_classification_data_dir, 'img_size': (512,512)}
    datapoint_class = dataset_classification.COVID_CT_DATA(**dataset_class_pars)
    dataloader_class_pars = {'shuffle': True, 'batch_size': 1}
    dataloader_class = data.DataLoader(datapoint_class, **dataloader_class_pars)
    # parameters for the segmentation dataset/dataloaders
    # 512x512 is the recommended image size input
    dataset_seg_pars = {'stage': 'train', 'gt': os.path.join(train_segmentation_data_dir, gt_dir),
                          'data': os.path.join(train_segmentation_data_dir, imgs_dir), 'mask_type': 'merge',
                          'ignore_small': True}
    datapoint_seg = dataset_segmentation.CovidCTData(**dataset_seg_pars)
    dataloader_seg_pars = {'shuffle': False, 'batch_size': 1}
    dataloader_seg = data.DataLoader(datapoint_seg, **dataloader_seg_pars)
    #############################################################################################
    # CREATE SSM
    # IF A PRETRAINED MODEL IS PROVIDED
    # This must be the full path to the checkpoint with the anchor generator and model weights
    # Assumed that the keys in the checkpoint are model_weights and anchor_generator
    if use_pretrained:
        ckpt = torch.load(checkpoint, map_location=device)
        sizes = ckpt['anchor_generator'].sizes
        aspect_ratios = ckpt['anchor_generator'].aspect_ratios
        anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    # or create from scratch
    else:
        anchor_generator = AnchorGenerator(sizes=((2, 4, 8, 16, 32),), aspect_ratios=((0.1, 0.25, 0.5, 1, 1.5, 2),))
    # keyword arguments
    # box_score_threshold:negative!
    # set RPN NMS thresholds to 0.75 to get adjacent predictions
    # RoI NMS is not used (set to -0.01)
    # Box_detection_per_img: RoI batch size for the image classification module
    ssm_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024,
                           'detections_per_img': box_detections, 'box_nms_thresh_classifier': box_nms_thresh_classifier,
                           'box_score_thresh_classifier': -0.01, 'rpn_nms_thresh': rpn_nms_th}
    # 256: number of channels in FPN, 2: number of segmentation classes (Lesions + Background), 3: number of image
    # classes (COVID+CP+Normal)
    box_head_input_size = 256 * 7 * 7
    box_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    box_head = TwoMLPHead(in_channels=box_head_input_size, representation_size=128)
    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=2)
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    mask_rcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
    mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=2)
    classifier_covid = S2Predictor(in_channels=5, batch_size=box_detections, representation_size=512,
                                   out_channels=3)
    # add arguments and create model
    ssm_args['rpn_anchor_generator'] = anchor_generator
    ssm_args['mask_head'] = mask_rcnn_heads
    ssm_args['box_roi_pool'] = box_roi_pool
    ssm_args['mask_roi_pool'] = mask_roi_pool
    ssm_args['mask_predictor'] = mask_predictor
    ssm_args['box_predictor'] = box_predictor
    ssm_args['box_head'] = box_head
    ssm_args['s2predictor'] = classifier_covid
    # This creates the truncated ResNet18+FPN backbone (last block deleted)
    
    ssm = maskrcnn_resnet_fpn(backbone_name='resnet18', pretrained=True, **ssm_args)
    if use_pretrained:
        ssm.load_state_dict(ckpt['model_weights'])
    print(ssm)
    if device == torch.device('cuda'):
        ssm = ssm.to(device)
    optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
    optimizer = torch.optim.Adam(ssm.parameters(), **optimizer_pars)
    if use_pretrained and 'optimizer_state' in ckpt.keys():
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if start_epoch>0:
        num_epochs += start_epoch
    print("Start training, epoch = {:d}".format(start_epoch))
    max_loss = 10000
    for e in range(start_epoch, num_epochs):
        loss_epoch_classification, loss_epoch_segmentation, max_loss = main_step("train", e, dataloader_class, dataloader_seg,
                                                                       optimizer, device, ssm, save_every, lrate,
                                                                       model_name, anchor_generator, save_dir, max_loss)
        print("Epoch {0:d}: train loss = {1:.3f}, validation loss = {2:.3f}".format(e,
             loss_epoch_classification, loss_epoch_segmentation))
    end_time = time.time()
    print("Training took {0:.1f} seconds".format(end_time - start_time))


#########################################################################
# MAIN LOOP FOR 1 EPOCH
def main_step(stage, e, dataloader_class, dataloader_seg, optimizer, device, model,
              save_every, lrate, model_name, anchors, save_dir, max_loss):
    epoch_loss_classification = 0
    epoch_loss_segmentation = 0
    total_segmentation_steps = 0
    for bt in dataloader_class:
        optimizer.zero_grad()
        # sample randomly from the segmentation dataset
        # no batch dimension!
        rand_im_segment_ind = torch.randint(len(dataloader_seg), size=(1,)).item()
        b = dataloader_seg.dataset[rand_im_segment_ind]
        X, y = b
        if device == torch.device('cuda'):
            X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y[
                'masks'].to(device)
        # no batch dimension!
        images = [X]
        targets = []
        lab = {}
        lab['boxes'], lab['labels'], lab['masks'] = y['boxes'], y['labels'], y['masks']
        targets.append(lab)
        # SEGMENTATION STEP #
        if len(targets[0]['boxes']):
            total_segmentation_steps += 1
            model.train()
            # output: segmentation loss only
            loss, _, _ = model(images, targets)
            total_seg_loss = 0
            for k in loss.keys():
                total_seg_loss += loss[k]
            total_seg_loss.backward()
            # Alex:
            # No update of the image classification module
            # This should not be necessary, as the classification branch is not used
            for _n, _p in model.named_parameters():
                if 's2' in _n:
                    if _p.grad is not None:
                        _p.grad = None
            epoch_loss_segmentation += total_seg_loss.item()
            # Alex:
            # only if better
            # copy weights from the segmentation
            # into the classification branch
            if total_seg_loss.item() < max_loss:
                utils.copy_weights(model)
                max_loss = total_seg_loss.item()
            optimizer.step()
            optimizer.zero_grad()
        # IMAGE CLASSIFICATION STEP
        # Alex:
        # set all to eval, then switch back parameters upgrade for S2 only!
        # This should not be necessary, as there are no batchnorm/dropout layers in S2
        # BatchNorm2D layers stay in eval mode!
        # compute the image loss
        model.eval()
        model.s2new.train()
        X_cl, y_cl = bt
        if device == torch.device('cuda'):
            X_cl, y_cl = X_cl.to(device), y_cl.to(device)
        image = [X_cl.squeeze_(0)]  # remove the batch dimension
        # Output: only the image class scores vector of 3
        _, predict_score, _ = model(image, targets=None)
        classifier_loss = F.binary_cross_entropy_with_logits(predict_score[0]['final_scores'], y_cl.squeeze_(0))
        classifier_loss.backward()
        # Alex
        # only upgrade the classification module+backbone for the best results
        # This excludes RPN, as the segmentation branch is not used
        # Assumes that the classifier module's name is s2new
        for _n, _p in model.named_parameters():
            if 's2new' not in _n and 'backbone' not in _n:
                _p.grad = None
        optimizer.step()
        epoch_loss_classification += classifier_loss.item()
    # save model?
    if not e % save_every:
        model.eval()
        state = {'epoch': str(e), 'model_weights': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'lrate': lrate, 'anchor_generator':anchors}
        if model_name is None:
            torch.save(state, os.path.join(save_dir, "ssm_ckpt_" + str(e+1) + ".pth"))
        else:
            torch.save(state, os.path.join(save_dir, model_name + "_ckpt_" + str(e+1) + ".pth"))
        model.train()
    epoch_loss_classification = epoch_loss_classification / len(dataloader_class)
    epoch_loss_segmentation = epoch_loss_segmentation / total_segmentation_steps
    return epoch_loss_classification, epoch_loss_segmentation, max_loss


# run the training
if __name__ == '__main__':
    config_train = config_ssm.get_config_pars_ssm("trainval")
    main(config_train, main_step)
