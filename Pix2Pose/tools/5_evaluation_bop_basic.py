import os,sys
from math import radians
import csv
import cv2
from skimage.transform import resize

from matplotlib import pyplot as plt
import time
import random
import numpy as np
import transforms3d as tf3d
from tools import myrender
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

if(len(sys.argv)!=4):
    print("python3 tools/5_evaluation_bop_basic.py [gpu_id] [cfg file] [dataset_name]")
    sys.exit()
    
gpu_id = sys.argv[1]
if(gpu_id=='-1'):
    gpu_id=''
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import tensorflow as tf
from bop_toolkit_lib import inout
from tools import bop_io
from pix2pose_util import data_io as dataio
from pix2pose_model import ae_model as ae
from pix2pose_model import recognition as recog
from pix2pose_util.common_util import get_bbox_from_mask

cfg_fn =sys.argv[2]
cfg = inout.load_json(cfg_fn)
detect_type = cfg['detection_pipeline'] #we use maskrcnn here
print("detect_type = ",detect_type)
# TODO: delete===>
detection_dir=cfg['path_to_detection_pipeline']
sys.path.append(detection_dir)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from tools.mask_rcnn_util import BopInferenceConfig
def get_rcnn_detection(image_t,model):
    image_t_resized, window, scale, padding, crop = utils.resize_image(
                    np.copy(image_t),
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
    if(scale!=1):
        print("Warning.. have to adjust the scale")
    results = model.detect([image_t_resized], verbose=0)
    r = results[0]
    rois = r['rois']
    if(scale!=1):
        masks_all = r['masks'][window[0]:window[2],window[1]:window[3],:]
        masks = np.zeros((image_t.shape[0],image_t.shape[1],masks_all.shape[2]),bool)
        for mask_id in range(masks_all.shape[2]):
            masks[:,:,mask_id]=resize(masks_all[:,:,mask_id].astype(np.float),(image_t.shape[0],image_t.shape[1]))>0.5
        #resize all the masks
        rois=rois/scale
        window = np.array(window)
        window[0] = window[0]/scale
        window[1] = window[1]/scale
        window[2] = window[2]/scale
        window[3] = window[3]/scale
    else:
        masks = r['masks'][window[0]:window[2],window[1]:window[3],:]

    rois = rois - [window[0],window[1],window[0],window[1]]
    obj_orders = np.array(r['class_ids'])-1
    obj_ids = model_ids[obj_orders]
    #now c_ids are the same annotation those of the names of ply/gt files
    scores = np.array(r['scores'])
    return rois,obj_orders,obj_ids,scores,masks
# TODO: delete<===

score_type = cfg["score_type"]
#1-scores from a 2D detetion pipeline is used (used for the paper)
#2-scores are caluclated using detection score+overlapped mask (only supported for Mask RCNN, sued for the BOP challenge)

task_type = cfg["task_type"]
#1-Output all results for target object in the given scene
#2-ViVo task (2019 BOP challenge format, take the top-n instances)
cand_factor =float(cfg['cand_factor'])

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_tf)

############################# Parse Dataset ##########################
dataset=sys.argv[3]
vis=False
output_dir = cfg['path_to_output']
output_img = output_dir+"/" +dataset
if not(os.path.exists(output_img)):
    os.makedirs(output_img)

bop_dir,test_dir,model_plys,\
model_info,model_ids,rgb_files,\
depth_files,mask_files,gts,\
cam_param_global,scene_cam = bop_io.get_dataset(cfg,dataset,incl_param=True,train=False) #TODO: check the paths here

im_width,im_height =cam_param_global['im_size'] 
cam_K = cam_param_global['K']
model_params =inout.load_json(os.path.join(bop_dir+"/models_xyz/",cfg['norm_factor_fn']))

if(dataset=='itodd'):
    img_type='gray'
else:
    img_type='rgb'
    

if("target_obj" in cfg.keys()):
    target_obj = cfg['target_obj']
    remove_obj_id=[]
    incl_obj_id=[]
    for m_id,model_id in enumerate(model_ids):
        if(model_id not in target_obj):
            remove_obj_id.append(m_id)
        else:
            incl_obj_id.append(m_id)
    for m_id in sorted(remove_obj_id,reverse=True):
        print("Remove a model:",model_ids[m_id])
        del model_plys[m_id]
        del model_info['{}'.format(model_ids[m_id])]
        
    model_ids = model_ids[incl_obj_id]
    

print("Camera info-----------------")
print(im_width,im_height)
print(cam_K)
print("-----------------")

'''
standard estimation parameter for pix2pose
'''

th_outlier=cfg['outlier_th']
dynamic_th = True
if(type(th_outlier[0])==list):
    print("Individual outlier thresholds are applied for each object")
    dynamic_th=False
    th_outliers = np.squeeze(np.array(th_outlier))
th_inlier=cfg['inlier_th']
th_ransac=3

dummy_run=True
MODEL_DIR = os.path.join(bop_dir, "weight_detection")
if(detect_type=='rcnn'):
    #Load mask r_cnn
    '''
    standard estimation parameter for Mask R-CNN (identical for all dataset)
    '''
    config = BopInferenceConfig(dataset=dataset,
                            num_classes=model_ids.shape[0]+1,#1+len(model_plys),
                            im_width=im_width,im_height=im_height)
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,
                            model_dir=MODEL_DIR)
    last_path = model.find_last()
    #Load the last model you trained and continue training
    model.load_weights(last_path, by_name=True)


'''
Load pix2pose inference weights
'''
load_partial=False
obj_pix2pose=[]
obj_names=[]
image_dummy=np.zeros((im_height,im_width,3),np.uint8)
if( 'backbone' in cfg.keys()):
    backbone = cfg['backbone']
else:
    backbone = 'paper'
# load model weights for each object
# obj_pix2pose[list]: contains several pix2pose() (from recognition.py) instances which are used to estimate the pose.
# obj_names: contian the id of object
obj_bboxes = [] #list of 3D bbox of 21 objects
for m_id,model_id in enumerate(model_ids):
    model_param = model_params['{}'.format(model_id)]
    obj_param=bop_io.get_model_params(model_param)
    weight_dir = bop_dir+"/pix2pose_weights/{:02d}".format(model_id)
    #weight_dir = "/home/kiru/media/hdd/weights/tless/tless_{:02d}".format(model_id)
    if(backbone=='resnet50'):
        #weight_fn = os.path.join(weight_dir,"inference_resnet_model.hdf5")
        weight_fn = os.path.join(weight_dir,"inference_resnet50.hdf5")
    else:
        weight_fn = os.path.join(weight_dir,"inference.hdf5")
    print("load pix2pose weight for obj_{} from".format(model_id),weight_fn)
    if not(dynamic_th):
        th_outlier = [th_outliers[m_id]] #provid a fixed outlier value
        print("Set outlier threshold to ",th_outlier[0])    
    recog_temp = recog.pix2pose(weight_fn,camK= cam_K,
                                res_x=im_width,res_y=im_height,obj_param=obj_param,
                                th_ransac=th_ransac,th_outlier=th_outlier,
                                th_inlier=th_inlier,backbone=backbone)
    obj_pix2pose.append(recog_temp)    
    obj_names.append(model_id)

    ply_fn = os.path.join(cfg['model_dir'],"obj_0000{:02d}.ply".format(model_id))
    obj_model = inout.load_ply(ply_fn)
    obj_bboxes.append(myrender.get_3d_box_points(obj_model['pts']))


######################################## MaskRCNN Detection #############################
test_target_fn = cfg['test_target']
target_list = bop_io.get_target_list(os.path.join(bop_dir,test_target_fn+".json")) # load test dataset information

prev_sid=-1 # avoid repeatedly load file for the same scene
result_dataset=[]

model_ids_list = model_ids.tolist()

# TODO:to activate networks, but Why???
if(dummy_run):
    #to activate networks before atual recognition
    #since only the first run takes longer time
    image_t = np.zeros((im_height,im_width,3),np.uint8)
    if(detect_type=='rcnn'):
        rois,obj_orders,obj_ids,scores,masks = get_rcnn_detection(image_t,model)
    elif(detect_type=='retinanet'):
        rois,obj_orders,obj_ids,scores = get_retinanet_detection(image_t,model)

# every scene has its own camera file
for scene_id,im_id,obj_id_targets,inst_counts in target_list:
    print("Recognizing scene_id:{}, im_id:{}".format(scene_id,im_id))
    if(prev_sid!=scene_id):
        cam_path = test_dir+"/{:06d}/scene_camera.json".format(scene_id)
        cam_info = inout.load_scene_camera(cam_path)
        if(dummy_run):
            image_t = np.zeros((im_height,im_width,3),np.uint8)        
            for obj_id_target in obj_id_targets: #refreshing
                # TODO: what has been done here???
                _,_,_,_,_,_ = obj_pix2pose[model_ids_list.index(obj_id_target)].est_pose(image_t,np.array([0,0,128,128],np.int))    
    
    prev_sid=scene_id #to avoid re-load scene_camera.json
    cam_param = cam_info[im_id]
    cam_K = cam_param['cam_K']
    depth_scale = cam_param['depth_scale'] #depth/1000 * depth_scale

    # TODO: load RGB image as <image_t>
    if(img_type=='gray'):
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.tif".format(im_id)
        image_gray = inout.load_im(rgb_path)
        #copy gray values to three channels    
        image_t = np.zeros((image_gray.shape[0],image_gray.shape[1],3),dtype=np.uint8)
        image_t[:,:,:]= np.expand_dims(image_gray,axis=2)
    else:
        rgb_path = test_dir+"/{:06d}/".format(scene_id)+img_type+\
                        "/{:06d}.png".format(im_id)
        image_t = inout.load_im(rgb_path) #load single narual image for detection

    t1=time.time()
    inst_count_est=np.zeros((len(inst_counts)))
    inst_count_pred = np.zeros((len(inst_counts))) # detected instances counter
    
    if(detect_type=='rcnn'):
        rois,obj_orders,obj_ids,scores,masks = get_rcnn_detection(image_t,model) #obj_orders is used to index the obj_pix2pose. The elements are 1 less than the obj_ids
    elif(detect_type=='retinanet'):
        rois,obj_orders,obj_ids,scores = get_retinanet_detection(image_t,model)

    # initialize the list to collect the results of pose estimation
    result_score=[]
    result_objid=[]
    result_R=[]
    result_t=[]
    result_bbox=[]
    # result_img=[]
    result_poses = []
    img_pose = np.copy(image_t)
    vis=False

    # estimate pose for each detected objects/roi
    for r_id,roi in enumerate(rois):
        if(roi[0]==-1 and roi[1]==-1):
            continue
        obj_id = obj_ids[r_id]        
        if not(obj_id in obj_id_targets):
            #skip if the detected object is not in the target object(ground truth)
            continue           
        obj_gt_no = obj_id_targets.index(obj_id)
        if(inst_count_pred[obj_gt_no]>inst_counts[obj_gt_no]*cand_factor): #If the number of detected instances is 2(can_factor) larger then the # given by the groundtruth, then we stop process this bbox
          continue
        inst_count_pred[obj_gt_no]+=1

        obj_order_id = obj_orders[r_id]
        obj_pix2pose[obj_order_id].camK=cam_K.reshape(3,3)

        #TODO: pose estimation
        img_pred,mask_pred,rot_pred,tra_pred,frac_inlier,bbox_t =\
        obj_pix2pose[obj_order_id].est_pose(image_t,roi.astype(np.int)) # TODO: Why pose estimation output mask_pred???

        if(frac_inlier==-1):
            continue        
        if(score_type==2 and detect_type=='rcnn'):       
            mask_from_detect = masks[:,:,r_id]         
            union = np.sum(np.logical_or(mask_from_detect,mask_pred))
            if(union<=0):
                mask_iou=0
            else:
                mask_iou = np.sum(np.logical_and(mask_from_detect,mask_pred))/union # TODO: Why compute iou between mask_pred and mask_from_detect?
            score=scores[r_id]*frac_inlier*mask_iou*union # TODO: What does this score mean?
        else:
            score = scores[r_id]        
        #inst_count_pred[obj_gt_no]+=1
        tra_pred = tra_pred * 0.001  # mm to m #TODO:why
        if (tra_pred[2] < 0.1 or tra_pred[2] > 5):
            print("tra_pred is in wrong scale!")
            continue
        pred_tf = np.eye(4)
        pred_tf[:3, :3] = rot_pred
        pred_tf[:3, 3] = tra_pred

        result_poses.append(pred_tf)
        result_score.append(score)
        result_objid.append(obj_id)
        result_R.append(rot_pred)
        result_t.append(tra_pred)
        result_bbox.append(roi)
        
    if len(result_score)>0:
        result_score = np.array(result_score)
        result_score = result_score/np.max(result_score) #normalize
        sorted_id = np.argsort(1-result_score) #sort results
        time_spend =time.time()-t1 #ends time for the computation
        total_inst=0
        n_inst = np.sum(inst_counts)
    else:
        continue

    # render pose estimation results
    for o_id, tf, score, roi in zip(result_objid, result_poses, result_score, result_bbox):
        img_pose = myrender.draw_3d_poses(obj_bboxes[o_id-1], tf, img_pose, cam_K.reshape(3, 3))
        cv2.putText(img_pose, '{:.3f}'.format(score), (roi[1], roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, 1)
        # print(o_id)
    cv2.imwrite(cfg["rendered_figures_dir"]+"/scene_id:{}_im_id:{}.jpg".format(scene_id,im_id), img_pose[:,:,::-1])
    # cv2.imshow("ww", img_pose[:,:,::-1])
    # cv2.waitKey(0)

    for result_id in sorted_id:
        total_inst+=1
        if(task_type=='2' and total_inst>n_inst): #for vivo task
            break        
        obj_id = result_objid[result_id]
        R = result_R[result_id].flatten()
        t = (result_t[result_id]).flatten()
        score = result_score[result_id]
        obj_gt_no = obj_id_targets.index(obj_id)
        inst_count_est[obj_gt_no]+=1
        if(task_type=='2' and inst_count_est[obj_gt_no]>inst_counts[obj_gt_no]):
            #skip if the result exceeds the amount of taget instances for vivo task
            continue
        #TODO: Write R,t into `result_temp`
        result_temp ={'scene_id':scene_id,'im_id': im_id,'obj_id':obj_id,'score':score,'R':R,'t':t,'time':time_spend }
        result_dataset.append(result_temp)





if(dataset=='tless'):
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test-primesense.csv")
else:
    output_path = os.path.join(output_dir,"pix2pose-iccv19_"+dataset+"-test.csv")

print("Saving the result to ",output_path)

#TODO: the result R and t are stored here
inout.save_bop_results(output_path,result_dataset)

