import cv2
import copy
import torch
import numpy as np
from torch.utils.data import Dataset

from core.config import cfg
from models.templates import smplh, obj_dict
from funcs_utils import load_img, get_bbox, batch_rodrigues, transform_joint_to_other_db
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smplh_param_processing, obj_param_processing, flip_joint, get_aug_config


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_human_2d = False
        self.has_human_3d = False
        self.has_smpl_param = False
        self.has_obj_param = False
        self.has_contact = True
        self.load_mask = True

    def __len__(self):
        # return 2048
        return len(self.datalist)
  
    def __getitem__(self, index):
        data = copy.deepcopy(self.datalist[index])

        # image
        img_path = data['img_path']
        try:
            img = load_img(img_path)
        except OSError:
            data = copy.deepcopy(self.datalist[min(index+1, self.__len__()-1)])
            img_path = data['img_path']
            img = load_img(img_path)

        bbox = data['bbox']
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, cfg.MODEL.input_img_shape, self.data_split, augment=False)

        # Post processing
        img = self.transform(img.astype(np.float32)/255.0)
    
        if self.load_mask:
            if self.joint_set['name'] == 'BEHAVE':
                human_mask_path = img_path.replace('.color.jpg', '.person_mask.jpg')
                obj_mask_path = img_path.replace('.color.jpg', '.obj_rend_mask.jpg')
            elif self.joint_set['name'] == 'InterCap':
                img_name = img_path.split('/')[-1].split('.')[0]
                human_mask_path = img_path.replace('/sequences', '/intercap_masks').replace('/color', '/mask').replace(f'{img_name}.jpg', f'{img_name}_human.png')
                obj_mask_path = img_path.replace('/sequences', '/intercap_masks').replace('/color', '/mask').replace(f'{img_name}.jpg', f'{img_name}_object.png')
            else:
                assert "Invalid joint set!"
            
            human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)
            human_mask = cv2.warpAffine(human_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
            human_mask[human_mask<128] = 0; human_mask[human_mask>=128] = 1
            
            obj_mask = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
            obj_mask = cv2.warpAffine(obj_mask, img2bb_trans, (cfg.MODEL.input_img_shape[1], cfg.MODEL.input_img_shape[0]), flags=cv2.INTER_LINEAR)
            obj_mask[obj_mask<128] = 0; obj_mask[obj_mask>=128] = 1

            human_mask, obj_mask = torch.tensor(human_mask).float(), torch.tensor(obj_mask).float()
            img = torch.cat((img, human_mask[None], obj_mask[None]))
        else:
            obj_mask = torch.zeros((cfg.MODEL.input_img_shape[0], cfg.MODEL.input_img_shape[1], 2))

        inputs = {'img': img, 'obj_id': obj_id}

        return inputs

    