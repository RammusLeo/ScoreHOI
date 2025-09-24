import os
import cv2
import numpy as np

def scale_bounding_box(orig_w, orig_h, new_w, new_h, bbox):
    """
    对原图像的bounding box进行缩放变换，得到新图像上的bounding box。
    
    参数:
    orig_w (int): 原图像的宽度
    orig_h (int): 原图像的高度
    new_w (int): 新图像的宽度
    new_h (int): 新图像的高度
    bbox (tuple): 原图像上的bounding box，格式为 (x_min, y_min, x_max, y_max)

    返回:
    tuple: 新图像上的bounding box，格式为 (x_min', y_min', x_max', y_max')
    """
    x_min, y_min, x_max, y_max = bbox

    # 计算缩放比例
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # 缩放bounding box的坐标
    x_min_new = x_min * scale_x
    y_min_new = y_min * scale_y
    x_max_new = x_max * scale_x
    y_max_new = y_max * scale_y

    return np.array([x_min_new, y_min_new, x_max_new, y_max_new])

def refine_bbox(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0], bbox[1], w, h])

def combine_masks(person_mask, object_mask):
    # 将两张mask结合，使用逻辑或操作
    combined_mask = np.logical_or(person_mask, object_mask)
    return combined_mask

def get_bounding_box(mask, margin=20):
    # 找到mask中所有白色像素的坐标
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # 计算正方形的边长
    side_length = max(ymax - ymin, xmax - xmin)

    # 计算正方形的中心
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # 计算正方形的边界，并加上margin
    xmin = max(center_x - side_length // 2 - margin, 0)
    xmax = min(center_x + side_length // 2 + margin, mask.shape[1] - 1)
    ymin = max(center_y - side_length // 2 - margin, 0)
    ymax = min(center_y + side_length // 2 + margin, mask.shape[0] - 1)

    return (xmin, ymin, xmax, ymax)
