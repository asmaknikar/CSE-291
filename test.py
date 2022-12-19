import datetime
import json
import multiprocessing
import os
import pickle
import warnings
from collections import defaultdict
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from operator import attrgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from skimage.transform import rescale, resize
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from tqdm import tqdm

import utils
from benchmark_utils import pose_utils
from UNet import Unet
from UNet2 import Unet2

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

point_cloud_dict = load_pickle("obj_pcd_dict.pickle")

def get_split_files(split_name):
    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        rgb = [p + "_color_kinect.png" for p in prefix]
        depth = [p + "_depth_kinect.png" for p in prefix]
        label = [p + "_label_kinect.png" for p in prefix]
        meta = [p + "_meta.pkl" for p in prefix]
    return rgb, depth, label, meta

def np2o3d(points_viewer):
    # points = open3d.utility.Vector3dVector(points_viewer.reshape([-1, 3]))
    # colors = open3d.utility.Vector3dVector(rgb.reshape([-1, 3]))
    points = open3d.utility.Vector3dVector(points_viewer.reshape([-1, 3]))
    pcd = open3d.geometry.PointCloud()
    pcd.points = points
    # pcd.colors = colors
    return pcd

def draw_bb(poses_world,meta,rgb):
    box_sizes = np.array([meta['extents'][idx] * meta['scales'][idx] for idx in meta['object_ids']])
    boxed_image = np.array(rgb)
    for i in range(len(poses_world)):
        utils.draw_projected_box3d(
            boxed_image, poses_world[i][:3,3], box_sizes[i], poses_world[i][:3, :3], meta['extrinsic'], meta['intrinsic'],
            thickness=2)
    # plt.plot(Image.fromarray((boxed_image * 255).astype(np.uint8)))
    return(Image.fromarray((boxed_image * 255).astype(np.uint8)))

def point_cloud_image_in_c(meta,depth):
    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    points_viewer = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    return(points_viewer)

def align_rotation_matrix(vec1, vec2):
    """get rotation matrix between two vectors using scipy
    vec1->source
    vec2->target
    """
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()

def rot_align_via_PCA(src,tgt):
    try:
        pca = PCA(n_components=3)
        pca.fit(src)
        eig_vec = pca.components_
        normal_src = eig_vec[2, :] 
        pca = PCA(n_components=3)
        pca.fit(tgt)
        eig_vec = pca.components_
        normal_tgt = eig_vec[2, :] 
        return(align_rotation_matrix(normal_src, normal_tgt))
    except:
        return(np.eye(3))
    

def get_best_transform(pcd,obj,skip = 64,train_red_factor = 1):
    registrationResults = []
    init_t = np.eye(4)
    for pcd_tr in tqdm(point_cloud_dict[obj][::train_red_factor],leave=False):
        pcd_tr = pcd_tr[::skip]
        # pcd = pcd[::skip]
        shift = (np.mean(pcd,0)-np.mean(pcd_tr,0))
        init_t[:3,3] = shift
        init_t[:3,:3] = rot_align_via_PCA(pcd_tr,pcd)
        tmp = open3d.pipelines.registration.registration_icp(np2o3d(pcd_tr),
                                                             np2o3d(pcd),
                                                     0.2,init_t)        
        registrationResults.append([len(tmp.correspondence_set),tmp.inlier_rmse,tmp.transformation])
    registrationResults = np.array(registrationResults,dtype=object)
    if(not(np.all(registrationResults[:,0]==0))):
        registrationResults = registrationResults[np.squeeze(np.argwhere(registrationResults[:,0]!=0))]
    registrationResults = registrationResults[np.argsort(-registrationResults[:,0])[:len(registrationResults)//5]]
    try:
        T = np.copy(registrationResults[np.argmin(registrationResults[:,1])][2])
    except:
        print(f'error {registrationResults}')
        T = np.eye(4)
    # T = finetune_icp_with_entire_pcd(pcd,obj,T)
    # print(T,registrationResults[np.argmin(registrationResults[:,1])],registrationResults)
    # T[:3,3]+=shift
    return(T)

def finetune_icp_with_entire_pcd(pcd,obj,T):
    entire_pcd = np.vstack(point_cloud_dict[obj])
    entire_pcd = entire_pcd[::len(entire_pcd)//1000000]
    tmp = open3d.pipelines.registration.registration_icp(np2o3d(entire_pcd),
                                                        np2o3d(pcd),
                                                        0.1,T)
    return tmp.transformation



class Dataset(torch.utils.data.Dataset):
    def __init__(self, rgb_files,label_files):
        super().__init__()
        self.image_names = rgb_files
        self.label_names = label_files

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        rgb = np.array(Image.open(self.image_names[idx]))
        rgb = rgb[::4,::4]
        img = (
          torch.tensor(rgb / 255.0, dtype=torch.float32)
          .permute(2, 0, 1)
          .contiguous()
        )
        if(not(self.label_names is None)):
            label = np.array(Image.open(self.label_names[idx]))
            label = label[::4,::4]
            lbl = torch.tensor(label, dtype=torch.int64).contiguous()
        else:
            lbl = None
        return {"image": img, "label": lbl}

def get_model(model_name,pth_name = "saved_model_2022_12_07T03_56_29.pth"):
    if (model_name=='UNet'):
        model_loaded = Unet()
        model_loaded.load_state_dict(torch.load(f'./weights/{pth_name}',map_location=device))
    else:
        model_loaded = Unet2()
        model_loaded.load_state_dict(torch.load(f'./weights/{pth_name}',map_location=device))
    model_loaded.to(device)
    return  model_loaded

def get_label_for_image(model_loaded,rgb,isCHW=True):
    model_loaded.eval()
    if(not isCHW):
        rgb = torch.tensor(rgb,dtype=torch.float32).permute(2,0,1).contiguous()
    label = np.argmax(np.transpose(np.squeeze(model_loaded(torch.unsqueeze(rgb,0).to(device)).cpu().detach().numpy(),0),(1,2,0)),-1)

def get_labels(model_loaded,rgb_test_files,model_name):
    dataset_test = Dataset(rgb_test_files,None)
    model_loaded.eval()
    labels_test = []
    for i in tqdm(range(len(dataset_test))):
        rgb = dataset_test[i]['image']
        if(model_name=="UNet"):
            label = np.argmax(np.transpose(np.squeeze(model_loaded(torch.unsqueeze(rgb,0).to(device)).cpu().detach().numpy(),0),(1,2,0)),-1)
            label_up = resize(label, (720,1280), order = 0,preserve_range=True)
        else:
            label = np.argmax(np.transpose(np.squeeze(model_loaded(torch.unsqueeze(rgb,0).to(device)).cpu().detach().numpy(),0),(1,2,0)),-1)
            label_up = label
        # print(label_up.shape)
        labels_test.append(label_up)
    return np.array(labels_test)

if __name__ == "__main__":
    testing_data_dir = "./data/testing_data_final/testing_data_final_filtered/testing_data/v2.2"
    split_dir = "./data/training_data/training_data_filtered/training_data/splits/v2"

    with open("./data/testing_data_final/testing_data_final_filtered/testing_data/test.txt",'r') as f:
        prefix = [os.path.join(testing_data_dir, line.strip()) for line in f if line.strip()]
        rgb_test_files = [p + "_color_kinect.png" for p in prefix]
        depth_test_files = [p + "_depth_kinect.png" for p in prefix]
        # label_test_files = [p + "_label_kinect.png" for p in prefix]
        meta_test_files= [p + "_meta.pkl" for p in prefix]
    
    test_ans = {}
    SKIP=1
    IMSHOW = True
    TRAIN_RED_FACTOR = 16
    MODEL = 'UNet2'
    WEIGHTS = './UNet2/UNet2_saved_model_2022_12_08T00_53_08_final.pth'
    MODEL = 'UNet'
    WEIGHTS = './saved_model_2022_12_07T03_56_29.pth'
    CHUNK_SIZE = 50
    test_size = len(rgb_test_files)

    model_loaded = get_model(MODEL,WEIGHTS)

    for i in range(0,test_size,CHUNK_SIZE):  
        print("chunk")
        labels_test = get_labels(model_loaded,rgb_test_files[i:min(test_size,i+CHUNK_SIZE)],MODEL)
        for meta_file,depth_file,label,rgb_file in tqdm(zip(np.array(meta_test_files)[i:min(test_size,i+CHUNK_SIZE)],
                                                        np.array(depth_test_files)[i:min(test_size,i+CHUNK_SIZE)],
                                                        # np.array(label_test_files),
                                                        labels_test,
                                                        np.array(rgb_test_files)[i:min(test_size,i+CHUNK_SIZE)]),total=len(np.array(meta_test_files)[i:min(test_size,i+CHUNK_SIZE)])):
            depth = np.array(Image.open(depth_file)) / 1000   # convert from mm to m
            meta = load_pickle(meta_file)
            rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
            pc_image = point_cloud_image_in_c(meta,depth)
            extrinsic = meta['extrinsic']
            extr_inv = np.linalg.inv(extrinsic)
            im_name = Path(meta_file).name.split('_')[0]
            test_ans[Path(meta_file).name.split('_')[0]] = {'poses_world':[None]*79}
            for objind in tqdm(meta['object_ids'],leave=False):
                seg_pcd = pc_image[label==objind]
                seg_pcd_world = seg_pcd@ extr_inv[:3, :3].T + extr_inv[:3, 3]
                T = get_best_transform(seg_pcd_world,objind,SKIP,TRAIN_RED_FACTOR)
                # T[:3,:3]*=-1
                test_ans[Path(meta_file).name.split('_')[0]]['poses_world'][objind] = T.tolist()
            plt.figure(figsize=(20,20))
            if(IMSHOW):
                plt.imshow(np.array(draw_bb(np.array([test_ans[im_name]['poses_world'][idx] for idx in meta['object_ids']]),meta,rgb)))
                plt.show()


    # labels_test = get_labels(get_model(MODEL,WEIGHTS),rgb_test_files[200:],MODEL)
    # model_loaded = get_model(MODEL,WEIGHTS)
    # for meta_file,depth_file,label,rgb_file in tqdm(zip(np.array(meta_test_files)[200:],
    #                                                 np.array(depth_test_files)[200:],
    #                                                 # np.array(label_test_files),
    #                                                 labels_test,
    #                                                 np.array(rgb_test_files)[200:]),total=len(np.array(meta_test_files)[200:])):
    #     depth = np.array(Image.open(depth_file)) / 1000   # convert from mm to m
    #     meta = load_pickle(meta_file)
    #     rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
    #     pc_image = point_cloud_image_in_c(meta,depth)
    #     extrinsic = meta['extrinsic']
    #     extr_inv = np.linalg.inv(extrinsic)
    #     im_name = Path(meta_file).name.split('_')[0]
    #     test_ans[Path(meta_file).name.split('_')[0]] = {'poses_world':[None]*79}
    #     for objind in tqdm(meta['object_ids'],leave=False):
    #         seg_pcd = pc_image[label==objind]
    #         seg_pcd_world = seg_pcd@ extr_inv[:3, :3].T + extr_inv[:3, 3]
    #         T = get_best_transform(seg_pcd_world,objind,SKIP,TRAIN_RED_FACTOR)
    #         # T[:3,:3]*=-1
    #         test_ans[Path(meta_file).name.split('_')[0]]['poses_world'][objind] = T.tolist()
    #     plt.figure(figsize=(20,20))
    #     if(IMSHOW):
    #         plt.imshow(np.array(draw_bb(np.array([test_ans[im_name]['poses_world'][idx] for idx in meta['object_ids']]),meta,rgb)))
    #         plt.show()


    with open(f'test/test_pred_UNET2.json', 'w') as f:
        json.dump(test_ans, f, ensure_ascii=False)


        
