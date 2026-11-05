"""A two-class Mask R-CNN. The checkpoint stores its image scale."""
import cv2
import torch
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(config, pretrained=False):
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=config.get("trainable_backbone_layers", 3))
    model = MaskRCNN(backbone, num_classes=91, min_size=config["min_size"],
                     max_size=config["max_size"], box_detections_per_img=5, box_score_thresh=0.001,
                     rpn_pre_nms_top_n_train=config.get("pre_nms", 2000),
                     rpn_pre_nms_top_n_test=config.get("pre_nms", 1000),
                     rpn_post_nms_top_n_train=config.get("post_nms", 2000),
                     rpn_post_nms_top_n_test=config.get("post_nms", 1000),
                     box_batch_size_per_image=config.get("roi_batch", 512))
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
    head = model.roi_heads
    head.box_predictor = FastRCNNPredictor(head.box_predictor.cls_score.in_features, 2)
    head.mask_predictor = MaskRCNNPredictor(head.mask_predictor.conv5_mask.in_channels, 256, 2)
    return model


def tensor_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if str(device) == "cpu":
        torch.set_num_threads(payload["config"].get("cpu_threads", 2))
    model = build_model(payload["config"])
    model.load_state_dict(payload["state"])
    return model.to(device).eval(), payload["score_threshold"]

def working_frame(frame):
    scale = min(1.0, 1024 / max(frame.shape[:2]))
    if scale == 1:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

