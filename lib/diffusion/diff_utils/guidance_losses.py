import torch
from .geometry import perspective_projection
from models.templates import smplh
import numpy as np
import matplotlib.pyplot as plt
from funcs_utils import rot6d_to_aa, rot6d_to_rotmat
import cv2
def gmof(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure error function.
    Args:
        x : Raw error signal
        sigma : Robustness hyperparameter
    Returns:
        torch.Tensor: Robust error signal
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def keypoint_fitting_loss(
    model_joints: torch.Tensor,
    camera_translation: torch.Tensor,
    joints_2d: torch.Tensor,
    joints_conf: torch.Tensor,
    camera_center: torch.Tensor,
    focal_length: torch.Tensor,
    img_size: torch.Tensor,
    sigma: float = 100.0,
) -> torch.Tensor:
    """
    Loss function for model fitting on 2D keypoints.
    Args:
        model_joints       (torch.Tensor) : Tensor of shape [B, NJ, 3] containing the SMPL 3D joint locations.
        camera_translation (torch.Tensor) : Tensor of shape [B, 3] containing the camera translation.
        joints_2d          (torch.Tensor) : Tensor of shape [B, N, 2] containing the target 2D joint locations.
        joints_conf        (torch.Tensor) : Tensor of shape [B, N, 1] containing the target 2D joint confidences.
        camera_center      (torch.Tensor) : Tensor of shape [B, 2] containing the camera center in pixels.
        focal_length       (torch.Tensor) : Tensor of shape [B, 2] containing focal length value in pixels.
        img_size           (torch.Tensor) : Tensor of shape [B, 2] containing the image size in pixels (height, width).
    Returns:
        torch.Tensor: Total loss value.
    """
    img_size_ = img_size.max(dim=-1)[0]
    # Heuristic for scaling data_weight with resolution used in SMPLify-X
    data_weight = (1000.0 / img_size_).reshape(-1, 1, 1).repeat(1, 1, 2)

    # Project 3D model joints
    projected_joints = perspective_projection(
        model_joints, camera_translation, focal_length, camera_center=camera_center
    )    
    projected_joints = projected_joints/img_size_.unsqueeze(-1).unsqueeze(-1) -0.5
    '''
    # debug    
    renderimg = renderimg.cpu().numpy().astype(np.uint8) # 将值为1的部分变为255
    bg = renderimg[0]
    projected_joints_render = ((projected_joints+0.5)*512).long()
    joints_2d_render = ((joints_2d+0.5)*512).long()
    # 将坐标点绘制到图像上
    for idx, point in enumerate(projected_joints_render[0]):
        # 绘制圆点，参数为(图像, 圆心位置, 半径, 颜色, 厚度)
        cv2.circle(bg, (point[0].item(), point[1].item()), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(bg, (joints_2d_render[0, idx, 0].item(), joints_2d_render[0, idx, 1].item()), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.imwrite('keypoints.png', bg)
    import pdb; pdb.set_trace()

    '''
    # Compute robust reprojection loss
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = (
        (data_weight**2) * (joints_conf**2) * reprojection_error
    ).sum(dim=(1, 2))

    return reprojection_loss


def multiview_loss(
    body_pose_6d: torch.Tensor, consistency_weight: float = 300.0
) -> torch.Tensor:
    """
    Loss function for multiple view refinement.
    Args:
        body_pose_6d : Tensor of shape (V, 23, 6) containing the 6D pose of V views of a person.
        consistency_weight : Pose consistency loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    mean_pose = body_pose_6d.mean(dim=0).unsqueeze(dim=0)
    pose_diff = ((body_pose_6d - mean_pose) ** 2).sum(dim=-1)
    consistency_loss = consistency_weight**2 * pose_diff.sum()
    total_loss = consistency_loss
    return total_loss


def smoothness_loss(pred_pose_6d: torch.Tensor) -> torch.Tensor:
    """
    Loss function for temporal smoothness.
    Args:
        pred_pose : Tensor of shape [N, 144] containing the 6D pose of N frames in a video.
    Returns:
        torch.Tensor : Total loss value.
    """
    pose_diff = ((pred_pose_6d[1:] - pred_pose_6d[:-1]) ** 2).sum(dim=-1)
    return pose_diff

def interaction_loss(
    obj_verts, human_verts, contact_h, penetration=False, contact_o = None, contact_o_f = None
):
    '''
    obj_verts   B, 64, 3
    contact_h   B, 431, 3
    contact_o   B, 64
    pose        B, 52, 6
    shape       B, 10
    obj_pose    B, 6
    obj_trans   B, 3
    '''
    dists = torch.cdist(human_verts, obj_verts) # B X 431 X 64 
    dists_h, _ = torch.min(dists, 2) # (B) X 431, B X 431
    dists_o, _ = torch.min(dists, 1) # (B) X 64, B X 64
    
    # calculate floor contact
    if contact_o_f is not None:
        all_verts= torch.cat([human_verts, obj_verts], 1) # B X 495 X 3
        floor_max = torch.max(all_verts[:, :, 1], 1)[0] # B
        dists_o_f = torch.abs(obj_verts[:, :, 1] - floor_max.unsqueeze(1)) # B X 64
        contact_labels_o_f = contact_o_f > 0.5

    # pred_contact_semantic = pred_clean_x[:, :, -4:-2] # BS X T X 2
    contact_labels_h = contact_h > 0.5
    contact_labels_o = contact_o > 0.5
    # contact_labels = contact_labels.reshape(bs, -1)[:, :2].detach() # (BS*T) X 2

    zero_target_h = torch.zeros_like(dists_h).to(dists_h.device)
    zero_target_o = torch.zeros_like(dists_o).to(dists_o.device)
    contact_threshold = 0.02 
    loss_interaction_h = torch.nn.functional.mse_loss(torch.maximum(dists_h*contact_labels_h-contact_threshold, zero_target_h), zero_target_h, reduction='none').sum(1)
    loss_interaction_o = torch.nn.functional.mse_loss(torch.maximum(dists_o*contact_labels_o-contact_threshold, zero_target_o), zero_target_o, reduction='none').sum(1)
    loss_interaction = loss_interaction_h + loss_interaction_o

    if contact_o_f is not None:
        zero_target_o_f = torch.zeros_like(dists_o_f).to(dists_o_f.device)
        loss_interaction_o_f = torch.nn.functional.mse_loss(torch.maximum(dists_o_f*contact_labels_o_f-contact_threshold, zero_target_o_f), zero_target_o_f, reduction='none').sum(1)
    else:
        loss_interaction_o_f = 0.0

    loss_collision = 0.0
    if penetration:
        pairwise_distance = torch.norm(obj_verts.unsqueeze(2) - human_verts.unsqueeze(1), dim=-1, p=2)
        distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)

        human_verts_all = smplh.upsample(human_verts)
        smplx_face = torch.tensor(smplh.faces.astype(np.int64), device=human_verts_all.device)
        smplx_face_vertices = human_verts_all[:, smplx_face] # <B, F, 3, 3>

        e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
        e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
        e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
        e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
        smplx_face_normal = torch.cross(e1, e2)  # <B, F, 3>
        # smplx_vertex_normals = torch.zeros(human_verts_all.shape, dtype=torch.float, device=human_verts_all.device) # <B, V, 3>
        smplx_vertex_normals = torch.zeros_like(human_verts_all)
        smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
        smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
        smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
        smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1) # <B, V, 3>
        sampled_smplx_vertex_normals = smplh.downsample(smplx_vertex_normals)   # B, 431 ,3
            
        closest_human_point = human_verts.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)
        query_to_surface = closest_human_point - obj_verts
        query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)

        closest_vertex_normals = sampled_smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))  # (B, O, 3)
        same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
        sdf = same_direction.sign() * distance_to_human    # (B, O)            
        loss_collision = torch.sum(sdf <= 0, dim=-1) / sdf.shape[-1]

    return loss_interaction, loss_collision, loss_interaction_o_f

def object_mask_loss(keypoints, mask, camera_translation, focal_length, camera_center):
    """
    计算物体关键点分布在mask中的准确度。
    当所有关键点都位于mask内时，损失为0。

    参数:
    - keypoints: 形状为 (n, 3)，关键点的坐标，假设已经是投影到2D平面上的 (x, y, z)。
    - mask: 形状为 (H, W)，二值化的图像分割mask，值为1表示物体的区域，值为0表示背景区域。
    - image_size: 图像大小 (H, W)，默认为 (256, 256)。

    返回:
    - 损失值，值越小越好，当所有关键点在mask内时，损失为0。
    """
    # 获取图像尺寸
    H, W = mask.shape[-2], mask.shape[-1]
    keypoints_2d = perspective_projection(
        keypoints, camera_translation, focal_length, camera_center=camera_center
    )    
    # 假设keypoints的形状为 (n, 3)，我们只关心前两个维度 (x, y)，第三个维度 z 可以忽略
    keypoints_2d = keypoints_2d.long()  # 获取二维坐标并转为整数
    bs = keypoints_2d.shape[0]
    '''    
    # debug
    # 如果图像是二值图像（0和1），将其转换为0和255的格式，以便绘制
    mask = (mask.cpu().numpy()*128).astype(np.uint8) # 将值为1的部分变为255
    mask_img = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2BGR)  # 将单通道图像转换为三通道图像
    # 将坐标点绘制到图像上
    for point in keypoints_2d[0]:
        # 绘制圆点，参数为(图像, 圆心位置, 半径, 颜色, 厚度)
        cv2.circle(mask_img, (point[0].item(), point[1].item()), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imwrite('mask.png', mask_img)
    import pdb; pdb.set_trace()
    '''
    # 确保关键点坐标在图像范围内
    keypoints_2d = torch.clamp(keypoints_2d, min=0, max=torch.tensor(W-1, dtype=torch.long))
    # B， 64， 2
    # 提取mask中对应的像素值
    # 注意：keypoints_2d[:, 0]是x坐标，keypoints_2d[:, 1]是y坐标
    # 使用keypoints的坐标从mask中获取对应位置的值，是否在物体区域内
    loss = 0
    for b in range(bs):
        mask_values = mask[b, keypoints_2d[b, :, 1], keypoints_2d[b, :, 0]]
        # 计算关键点在mask内的比例
        accuracy = torch.sum(mask_values) / mask_values.size(0)
        # 损失函数: 1 - 关键点在mask内的比例
        loss += 1 - accuracy
    
    return loss/bs