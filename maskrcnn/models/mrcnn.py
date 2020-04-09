from .detection.faster_rcnn import FastRCNNPredictor
from .detection.mask_rcnn import MaskRCNNPredictor
from .detection.rpn import AnchorGenerator
from .detection import maskrcnn_resnet50_fpn,MaskRCNN
from . import resnet
from . import senet
from torchvision.ops import misc as misc_nn_ops
from .detection.backbone_utils import BackboneWithFPN
def free_backbone(model):
    #freeze backbone until layer4
    for name, parameter in model.backbone.body.named_parameters():
        if 'layer4' not in name:
            #print(name)
            parameter.requires_grad_(False)


def get_instance_segmentation_model(num_classes,min_dim,max_dim,freeze_body=True,**kwargs):
    # load an instance segmentation model pre-trained on COCO
    # change anchor size to paper config
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    small_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
    model = maskrcnn_resnet50_fpn(pretrained=True,
                                  rpn_anchor_generator=small_anchor_generator,
                                  min_size=min_dim,
                                  max_size=max_dim,
                                  #overwrite box parameter,
                                  box_batch_size_per_image=128,
                                  **kwargs)
    #replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Replace Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    if freeze_body:
        free_backbone(model)
        print("Resnet layer0-3 is frozen.")
    return model


def get_maskrcnn_model(backbone,num_classes,min_dim,max_dim,freeze_body=True,freeze_bn=True,pretrained=True,**kwargs):
    if "se" in backbone:
        backbone = senet.__dict__[backbone](num_classes=1000, pretrained='imagenet' if pretrained else None,
                                          norm_layer=misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None)
    else:
        backbone = resnet.__dict__[backbone](pretrained=pretrained,
                                                  norm_layer=misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None)
    if pretrained:
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    backbone=BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    small_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
	

    model = MaskRCNN(backbone, 
                     num_classes=num_classes,
                     min_size=min_dim, 
                     max_size=max_dim,
                     rpn_anchor_generator=small_anchor_generator,
                     box_batch_size_per_image=128,
                     **kwargs)


    if freeze_body:
        free_backbone(model)
        print("Resnet layer0-3 is frozen.")
    return model
