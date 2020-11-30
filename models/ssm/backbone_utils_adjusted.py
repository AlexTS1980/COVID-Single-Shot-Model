
from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
# Alex: use only the last 
from .._utils import IntermediateLayerGetter
from .backbones import resnet


class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        # truncate the model to ooutput return_layers OrderedDict
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # body = truncated backbone
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            #extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        self.out_channels = out_channels


# out_channels=256
def resnet_fpn_backbone(backbone_name, pretrained, out_ch):
    print(backbone_name)
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=nn.BatchNorm2d)
    # This is for ResNet18, outputs the penultimate module
    # only use the last layer for FPN
    return_layers = {'layer3':'0'}
    in_channels_list = [backbone.out_channels]
    # 256
    out_channels = out_ch
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
