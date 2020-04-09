import torch
import numpy as np
from maskrcnn.models.mrcnn import get_maskrcnn_model
import json
__all__=['get_model_rcnn','get_rcnn_detection']
def get_model_rcnn(config_path):
    with open(config_path,"r") as f:
        config=json.load(f)
    model=get_maskrcnn_model(backbone=config['backbone'],
                             num_classes=config['num_classes'],
                             min_dim=config['min_dim'],
                             max_dim=config['max_dim'],
                             box_score_thresh=config['box_score_thresh'])
    model=model.cuda()
    model.load_state_dict(torch.load(config['model_path'])['state_dict'])
    model.eval()
    return model

def get_rcnn_detection(image,model):
    assert image.shape[2]==3
    image=image.astype(np.float32)/255.0
    image=torch.tensor(image,dtype=torch.float).permute(2,0,1).cuda()
    image=[image]
    with torch.no_grad():
        output=model(image)[0]
    bboxes=output['boxes'].detach().cpu().numpy()
    labels=output['labels'].detach().cpu().numpy() #1-21
    scores=output['scores'].detach().cpu().numpy()
    masks=output['masks'].detach().squeeze(1).cpu().numpy().transpose(1,2,0) > 0.9
    # exchange xy
    bboxes = bboxes[:,np.array([1,0,3,2])].astype(np.int)
    return bboxes, labels-1, labels, scores, masks