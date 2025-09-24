import cv2
import glob
import os
import copy
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from core.config import cfg
from models.templates import smplh, obj_dict
from funcs_utils import load_img, get_bbox, batch_rodrigues, transform_joint_to_other_db
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smplh_param_processing, obj_param_processing, flip_joint, get_aug_config


class DemoDataset(Dataset):
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.data_split = 'test'
        self.joint_set = {
            'name': 'BEHAVE',
            'joint_num': 73,
            'joints_name': (
                'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
                'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
                'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
                'L_BigToe', 'L_SmallToe', 'L_Heel', 'R_BigToe',  'R_SmallToe', 'R_Heel', 'L_Thumb_4', 'L_Index_4', 'L_Middle_4', 'L_Ring_4', 'L_Pinky_4', 'R_Thumb_4', 'R_Index_4', 'R_Middle_4', 'R_Ring_4', 'R_Pinky_4', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
                ),
            'flip_pairs': (
                (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48), (34, 49), (35, 50), (36, 51),
                (52, 55), (53, 56), (54, 57), (58, 63), (59, 64), (60, 65), (61, 66), (62, 67), (69, 70), (71, 72)
                ),
            'skeleton': (
                (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19), (19, 21), (9, 13), (13, 16), (16, 18), (18, 20), (9, 12), (12, 15),
                (20, 22), (22, 23), (23, 24), (20, 25), (25, 26), (26, 27), (20, 28), (28, 29), (29, 30), (20, 31), (31, 32), (32, 33), (20, 34), (34, 35), (35, 36),
                (21, 37), (37, 38), (38, 39), (21, 40), (40, 41), (41, 42), (21, 43), (43, 44), (44, 45), (21, 46), (46, 47), (47, 48), (21, 49), (49, 50), (50, 51),
                (7, 52), (7, 53), (7, 54), (8, 55), (8, 56), (8, 57), (36, 58), (24, 59), (27, 60), (33, 61), (30, 62), (51, 63), (39, 64), (42, 65), (48, 66), (45, 67), (12, 68), (68, 69), (68, 70), (69, 71), (70, 72)
            )
        }
        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')
        self.rgb_image_root = "data/COCO_2017/pick_images"
        self.mask_root = "data/COCO_2017/masks"
        self.objects_root = "data/COCO_2017/objects"
        self.load_mask = True

        # 获取所有图片文件
        catogory = sorted(os.listdir(self.rgb_image_root))
        self.datalist = []

        for cat in catogory:
            if cat == "skateboard":
                obj_name = "02"
            elif cat == "tennis_racket":
                obj_name = "05"
            else:
                obj_name = cat
            img_paths = glob.glob(os.path.join(self.rgb_image_root, cat, "*.jpg"))

            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                human_mask_path = img_path.replace('.jpg', '_hum.png').replace('pick_images', 'masks')
                obj_mask_path = img_path.replace('.jpg', '_obj.png').replace('pick_images', 'masks')

                data_dict = {
                    'img_path': img_path,
                    'human_mask_path': human_mask_path,
                    'obj_mask_path': obj_mask_path,
                    'obj_name': obj_name,
                    'ann_id': img_name.split('.')[0]  # 使用图片名作为标识符
                }
                self.datalist.append(data_dict)
        with open('data/base_data/object_models/behave/_info_feature.pkl', 'rb') as f:
            self.behave_objfeat = pickle.load(f)
        with open('data/base_data/object_models/intercap/_info_feature.pkl', 'rb') as f:
            self.intercap_objfeat = pickle.load(f)        
        
        self.objfeat_dict = {**self.behave_objfeat, **self.intercap_objfeat}

    def __len__(self):
        # return 2048
        return len(self.datalist)
  
    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])

        # image
        img_path = data['img_path']
        img = load_img(img_path)    

        human_mask_path = data['human_mask_path']
        obj_mask_path = data['obj_mask_path']
        human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)

        bbox = np.array(self.get_bbox(human_mask, obj_mask), dtype=np.float32)

        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, cfg.MODEL.input_img_shape, self.data_split, augment=False, force_flip=False, force_scale=1.0)

        # Post processing
        img = self.transform(img.astype(np.float32)/255.0)

        obj_name = data['obj_name']
        obj_verts_orig = obj_dict[obj_name].load_verts()

        obj_feat = obj_dict[obj_name].load_feat()
        if obj_feat is None:
            obj_feat = self.objfeat_dict[obj_name]
        obj_feat = np.array(obj_feat, dtype=np.float32)
        obj_feat = np.concatenate((obj_verts_orig.reshape(-1), obj_feat), -1)

        human_mask = cv2.warpAffine(human_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
        human_mask[human_mask<128] = 0; human_mask[human_mask>=128] = 1
        obj_mask = cv2.warpAffine(obj_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
        obj_mask[obj_mask<128] = 0; obj_mask[obj_mask>=128] = 1
        human_mask, obj_mask = torch.tensor(human_mask).float(), torch.tensor(obj_mask).float()
        img = torch.cat((img, human_mask[None], obj_mask[None]))

        if self.data_split == 'train':
            assert False, "Invalid usage!"
        
        inputs = {'img': img, 'obj_feat': obj_feat, 'obj_verts_orig': obj_verts_orig}
        targets = {}
        meta_info = {'ann_id': data['ann_id'], 'bbox':bbox, 'img2bb_trans': img2bb_trans, 'bb2img_trans': bb2img_trans, 'img_path': img_path, 'obj_name': obj_name, 'gender': 'neutral', "obj_mask": obj_mask}

        return inputs, targets, meta_info

    def get_bbox(self, human_mask, obj_mask):
        def get_largest_component(mask):
            # 将mask转换为uint8类型
            mask = mask.astype(np.uint8) * 255
            
            # 连通区域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            # 如果没有前景区域，返回空mask
            if num_labels < 2:
                return np.zeros_like(mask)
                
            # 找到面积最大的区域（跳过背景0）
            max_area = 0
            max_index = 1
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_index = i
            
            # 只保留最大的连通区域
            largest_component = np.zeros_like(mask)
            largest_component[labels == max_index] = 255
            return largest_component
        
        # 分别获取人体和物体的最大连通区域
        human_largest = get_largest_component(human_mask)
        obj_largest = get_largest_component(obj_mask)
        
        # 合并两个最大连通区域
        combined_mask = cv2.bitwise_or(human_largest, obj_largest)
        
        # 获取合并后mask的边界框
        y_indices, x_indices = np.where(combined_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No foreground object found in the combined mask.")
            
        x = np.min(x_indices)
        y = np.min(y_indices)
        w = np.max(x_indices) - x + 1
        h = np.max(y_indices) - y + 1
        
        # 计算中心点
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 取最大边长作为正方形边长
        side_length = max(w, h)
        
        # 扩大边长以达到1.8倍面积 (√1.8 ≈ 1.34)
        side_length = int(side_length * 1.3)
        
        # 从中心点计算新的bbox坐标
        half_side = side_length / 2
        x = int(center_x - half_side)
        y = int(center_y - half_side)
        
        # 返回正方形边界框参数（左上角x, y，宽w, 高h）
        bbox = (x, y, side_length, side_length)
        return bbox

'''
1. debug bbox & segmentation, bbox应该要尽量大
2. bbox应该是正方形
根据12获得标准的正方形输入图像及mask

'''