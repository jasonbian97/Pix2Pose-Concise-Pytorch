import torch
import numpy as np
import segmentation_models_pytorch as smp
import json

# config=dict()
# config['backbone']="resnet34"
# config['weights']=[
#     "unet/weights/ycbv_model_1_last.pth",
#     "unet/weights/ycbv_model_2_last.pth",
#     "unet/weights/ycbv_model_3_last.pth",
#    "unet/weights/ycbv_model_4_epoch15.pth",
#    "unet/weights/ycbv_model_5_epoch15.pth",
#    "unet/weights/ycbv_model_6_epoch15.pth",
#    "unet/weights/ycbv_model_7_epoch15.pth",
#    "unet/weights/ycbv_model_8_epoch15.pth",
#    "unet/weights/ycbv_model_9_epoch15.pth",
#    "unet/weights/ycbv_model_10_epoch15.pth",
#    "unet/weights/ycbv_model_11_epoch15.pth",
#   "unet/weights/ycbv_model_12_epoch15.pth",
#    "unet/weights/ycbv_model_13_epoch15.pth",
#    "unet/weights/ycbv_model_14_epoch15.pth",
#    "unet/weights/ycbv_model_15_epoch15.pth",
#    "unet/weights/ycbv_model_16_epoch15.pth",
#    "unet/weights/ycbv_model_17_epoch15.pth",
#    "unet/weights/ycbv_model_18_epoch15.pth",
#    "unet/weights/ycbv_model_19_epoch15.pth",
#    "unet/weights/ycbv_model_20_epoch15.pth",
#    "unet/weights/ycbv_model_21_epoch15.pth",
# ]
# with open("./unet_config.json", "w+") as f:
#     json.dump(config,f)

__all__=['get_modellist_unet','get_unet_pred']

def get_modellist_unet(config_path):
    modellist=[]
    with open(config_path,"r") as f:
        config=json.load(f)
    for p in config['weights']:
        model=smp.Unet(encoder_name=config['backbone'],classes=4)
        model = model.cuda()
        model.load_state_dict(torch.load(p)['generator'])
        model.eval()
        modellist.append(model)
    return modellist

def get_unet_pred(image,model):
    # assert image.shape==(128,128,3)
    # image=image.astype(np.float32)/255.0
    # image=(image-0.5)/0.5
    image=torch.tensor(image,dtype=torch.float).permute(2,0,1).unsqueeze(0).cuda()
    I_3d,I_e=torch.split(model(image),[3,1],dim=1)
    I_3d = torch.tanh(I_3d).detach().cpu().numpy().squeeze().transpose(1,2,0)#(128,128,3)
    I_e = torch.sigmoid(I_e).squeeze().detach().cpu().numpy()[:,:,np.newaxis] #(128,128,1)
    return I_3d,I_e