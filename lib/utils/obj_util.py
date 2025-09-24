import os
import cv2
import glob
import numpy as np
import torch
import pickle
import joblib
import trimesh
from pointnext.get_model import prepare_model 
from scipy.spatial.transform import Rotation as R

from bps_torch.bps import bps_torch
import json
class BPS():
    def __init__(self):
        self.bps = bps_torch(bps_type='random_uniform',
                        n_bps_points=64,
                        radius=1.,
                        n_dims=3,
                        custom_basis=None)

    def process(self, pc):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pointcloud = torch.tensor(pc).to(device)
        bps_enc = self.bps.encode(pointcloud,
                        feature_type=['dists','deltas'],
                        x_features=None,
                        custom_basis=None)
        deltas = bps_enc['deltas']
        bps_dec = self.bps.decode(deltas)
        bps_dec = bps_dec.squeeze(0).cpu().numpy()
        return bps_dec

class ObjProcessor():
    def __init__(self):
        self.bps = BPS()
        self.pointnext_model = prepare_model().cuda()

    def normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def load_obj(self, filename):
        return trimesh.load(filename, process=False, maintain_order=True, force='mesh')

    def obj2bps(self, filename):
        obj = self.load_obj(filename)
        verts = obj.vertices
        # verts = self.normalize(verts)
        # faces = obj.faces
        obj_bps = self.bps.process(verts)
        return obj_bps

    def bps2feat(self, bps):
        bps = torch.tensor(bps).float().cuda()
        bps = bps.unsqueeze(0)
        out = self.pointnext_model.get_feature(bps)
        feat = out.cpu().detach().numpy().squeeze(0)
        return feat

    def obj2feat(self, filename):
        bps = self.obj2bps(filename)
        feat = self.bps2feat(bps)
        return feat, bps

    def save_json(self, dataroot, save_path):
        data = {}
        for obj_name in sorted(os.listdir(dataroot)):
            obj_dir = os.path.join(dataroot, obj_name)
            obj_file = glob.glob(os.path.join(obj_dir, '*.obj'))[0]
            feat, bps = self.obj2feat(obj_file)
            data[obj_name] = {"path": obj_file, "feat": feat.tolist(), "kps": bps.tolist()}
        with open(save_path, 'w') as f:
            json.dump(data, f)

    def process_obj(self, obj_root):
        for obj_name in sorted(os.listdir(obj_root)):
            obj_dir = os.path.join(obj_root, obj_name)
            obj_file = glob.glob(os.path.join(obj_dir, '*.obj'))[0]
            obj = self.load_obj(obj_file)
            verts = obj.vertices
            verts = self.normalize(verts)
            new_obj = trimesh.Trimesh(vertices=verts, faces=obj.faces)
            basename = os.path.basename(obj_file)
            newname = 'new' + basename
            new_path = os.path.join(os.path.dirname(obj_file),newname)
            trimesh.exchange.export.export_mesh(new_obj, new_path, file_type='obj')

    def upscale_obj(self, obj_file):
        """将物体大小缩放为原来的一半并保存
        Args:
            obj: trimesh对象或obj文件路径
        Returns:
            new_obj: 缩放后的trimesh对象
        """
        # 如果输入是文件路径，先加载obj文件
        if isinstance(obj_file, str):
            obj = self.load_obj(obj_file)
        
        # 获取顶点和面
        verts = obj.vertices
        faces = obj.faces
        
        # 将顶点坐标缩放为原来的一半
        scaled_verts = verts * 1.5
        
        # 创建新的mesh对象
        new_obj = trimesh.Trimesh(vertices=scaled_verts, faces=faces)
        trimesh.exchange.export.export_mesh(new_obj, obj_file, file_type='obj')