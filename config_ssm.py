#SINGLE SHOT MODEL CONFIG
import argparse
import utils

# trainval
# test: get inference on new data
# precision: compute MS COCO mean average precision
def get_config_pars_ssm(stage):
    parser_ = argparse.ArgumentParser(
        description='arguments for training a Single Shot Model Lesion Segmentation/COVID-19 Prediction')

    parser_.add_argument("--device", type=str, default='cpu')
    parser_.add_argument("--model_name", type=str, default=None)
    parser_.add_argument("--rpn_nms_th", type=float, default=0.75, help="Both at train and test stages.")
    parser_.add_argument("--box_detections", type=int, default=128, help="Roi output batch, for the classification module.")
    parser_.add_argument("--ckpt", type=str, default=None,
                             help="Checkpoint file in .pth format. "
                                  "Must contain the following keys: model_weights, optimizer_state, anchor_generator")
    parser_.add_argument("--box_nms_thresh_classifier", type=float, default=0.75, help="NMS threshold for the classification prediction.")


    if stage == "trainval":
        parser_.add_argument("--start_epoch", type=str, default=0)
        parser_.add_argument("--checkpoint", type=str, help="Pretrained model, must be a checkpoint with keys:"
                                                       "model_weights, anchor_generator, optimizer_state, model_name",
                             default=None)
        parser_.add_argument("--num_epochs", type=int, default=50)
        parser_.add_argument("--save_dir", type=str, default="saved_models",
                             help="Directory to save checkpoints")
        parser_.add_argument("--train_seg_data_dir", type=str, default='../covid_data/train',
                             help="Path to the training data. Must contain --imgs_dir and --gt_dir.")
        parser_.add_argument("--gt_dir", type=str, default='masks',
                             help="Path to directory with binary masks. Must be in the seg data directory.")
        parser_.add_argument("--train_class_data_dir", type=str, default='../covid_data/cncb/train_large',
                             help="Path to the training data for the classification branch. Assumes that images names contain class ids.")
        parser_.add_argument("--imgs_dir", type=str, default='imgs', help="Dir with images for the segmentation problem. Must be in the seg data dir.")
        parser_.add_argument("--save_every", type=int, default=10)
        parser_.add_argument("--lrate", type=float, default=1e-5, help="Learning rate")


    elif stage == "test_segmentation" or stage == "inference_ssm":
        parser_.add_argument("--test_data_dir", type=str, default='../covid_data/test',
                             help="Path to the test data. Must contain images and may contain binary masks")
        parser_.add_argument("--imgs_dir", type=str, default='imgs', help="Directory with images. "
                                                                               "Must be in the test data directory.")


        parser_.add_argument("--confidence_th", type=float, default=0.75, help="Confidence score threshold on "
                                                                               "positive prediction from RoI.")
        parser_.add_argument("--mask_logits_th", type=float, default=0.5, help="Threshold for positve mask "
                                                                               "prediction at "
                                                                               "pixel level.")
        parser_.add_argument("--gt_dir", type=str, default='masks',
                             help="Path to directory with binary masks. Must be in the data directory.")
        parser_.add_argument("--roi_nms_th", type=float, default=0.5, help="NMS threshold for the segmentation prediction, Only at test stage.")

    elif stage == "test_classification" or stage == "inference_ssm":
        parser_.add_argument("--test_data_dir", type=str, default='../covid_data/test',
                             help="Path to the test data.")

    model_args = parser_.parse_args()
    return model_args
