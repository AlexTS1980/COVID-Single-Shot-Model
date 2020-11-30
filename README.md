# COVID-Single-Shot-Model Project

Conceptually the model is similar to COVID-CT-Mask-Net, but there are a lot of new functionality, so I decided to create a new repository. Of the models presented in the paper, I uploaded the architecture and the weights for the one trained from scratch with two parallel branches (segmentation/classification).

# Single Shot Model:

## The model
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Single-Shot-Model/blob/master/figures/ssm.png" width="800" height="300" align="center"/>
</p>


## Region of Interest (RoI) module with two branches
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Single-Shot-Model/blob/master/figures/roi_two_modules.png" width="800" height="400" align="center"/>
</p>

## RoI Batch To Feature Vector
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Single-Shot-Model/blob/master/figures/s_module_final.png" width="700" height="300" align="center"/>
</p>

Download the [pretrained weights](https://drive.google.com/file/d/1llKwQc0-1X70SgO1P9QlNqxP0vRXw6x_/view?usp=sharing) into a directory called `pretrained_weights`. The model uses ResNet18+FPN without the last block, to reduce the number of weights. To evaluate the model on the segmentation test split: 
```
python3 evaluation_seg_branch.py --ckpt pretrained_weights/modelA.pth --rpn_nms_th 0.75 
--roi_nms_th 0.5 --confidence_th 0.75 --device cuda --box_detections 128
```
You should get these results: 

|  Model	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| Model size
|:-:	|:-:	|:-:	|:-:|:-:	
| **SSM (ResNet18+FPN)** 	| 57.99% 	| 38.28% 	| 42.45% 	| 8.27M|

To evaluate the classification branch:
```
python3.5 evaluation_class_branch.py --ckpt pretrained_weights/modelA.pth --test_data path/to/test/data/ 
--device cuda  --box_nms_thresh_classifier 0.75 --box_detections 128
```
You should get this confusion matrix, COVID-19 sensitivity of **93.16%**, F1 score of **96.76%**.
|  	| Control 	| CP 	| COVID-19 	|
|:-:	|:-:	|:-:	|:-:	|
| **Control** 	| 9322 	| 123 	| 5 	|
| **CP** 	| 174 	| 7139 	| 82 	|
| **COVID-19** 	| 27 	| 277 	| 4042 	|

To train from scratch, make sure you have a directory `--train_seg_data_dir` with `--imgs_dir` and `--gt_dir` subdirectories for the segmentation branch and `--train_class_data_dir` for the classification branch. The links to the data  are here: http://ncov-ai.big.ac.cn/download, the train/test/validation splits are in `.txt` files above and in the source split: `https://github.com/haydengunraj/COVIDNet-CT/blob/master/docs/dataset.md`. 

On  a GPU with 8Gb VRAM 50 epochs should take about 5 hours. 
