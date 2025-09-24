import numpy as np
import time  # 添加这行导入
from typing import Dict, Tuple
from yacs.config import CfgNode
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from functools import partial

from diffusion.diff_utils.utils import *
from utils.funcs_utils import rot6d_to_rotmat, rot6d_to_aa, rotmat_to_6d, batch_rodrigues
# from diffusion.diff_utils.geometry import aa_to_rotmat, rot6d_to_rotmat, rot6d_to_aa, rotmat_to_6d
from diffusion.diff_utils.guidance_losses import keypoint_fitting_loss, multiview_loss, smoothness_loss, interaction_loss, object_mask_loss

import copy
import pickle
from models.templates import smplh

class GaussianDiffusionBetas(nn.Module):
    """ Class for the Diffusion Process (forward process and sampling methods). """

    def __init__(self, cfg: CfgNode, model: nn.Module, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = kwargs['device']

        ### Diffusion Proccess ###

        timesteps = cfg.DIFFUSION.DIFFUSION_PROCESS.timesteps
        beta_schedule = cfg.DIFFUSION.DIFFUSION_PROCESS.beta_schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # Helper function to register buffer from float64 to float32.
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For q(x_t | x_{t-1}) and others.
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # For posterior q(x_{t-1} | x_t, x_0).
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        p2_loss_weight_gamma = 0.0
        p2_loss_weight_k = 1.0
        register_buffer("p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)


        ## Denoising model
        self.model = model.to(self.device)
        self.use_betas = self.model.use_betas
        if self.use_betas:
            betas_stats = np.load(cfg.DIFFUSION.beta_stats, allow_pickle=True)
            self.betas_min = torch.from_numpy(betas_stats["betas_min"]).to(self.device).float()
            self.betas_max = torch.from_numpy(betas_stats["betas_max"]).to(self.device).float()
        self.loss_type = cfg.DIFFUSION.DENOISING_MODEL.loss_type
        self.objective = cfg.DIFFUSION.DENOISING_MODEL.objective
        assert self.objective in {"pred_noise", "pred_x0"}, "must be pred_noise or pred_x0"

        ## Sampling
        self.use_guidance = True
        if self.use_guidance:
            # Score guidance.
            self.optim_iters = kwargs.get("optim_iters", cfg.DIFFUSION.GUIDANCE.optim_iters)
            self.sample_start = kwargs.get("sample_start", cfg.DIFFUSION.GUIDANCE.sample_start)
            self.ddim_step_size = kwargs.get("ddim_step_size", cfg.DIFFUSION.GUIDANCE.ddim_step_size)
            self.grad_scale = kwargs.get("grad_scale", cfg.DIFFUSION.GUIDANCE.gradient_scale)
            self.use_hips = kwargs.get("use_hips", cfg.DIFFUSION.GUIDANCE.use_hips)
            self.early_stop = kwargs.get("early_stopping", cfg.DIFFUSION.GUIDANCE.earlystop)
            self.interaction_guidance = kwargs.get("interaction_guidance", cfg.DIFFUSION.GUIDANCE.w_inter>0)
            self.keypoint_guidance = kwargs.get("keypoint_guidance", cfg.DIFFUSION.GUIDANCE.w_kp2d>0)
            self.penetration_guidance = kwargs.get("penetration_guidance", cfg.DIFFUSION.GUIDANCE.w_colli>0)
            self.omask_guidance = kwargs.get("omask_guidance", cfg.DIFFUSION.GUIDANCE.w_omask>0)
            # self.multiview_guidance = kwargs.get("multiview_guidance", False)
            # self.temporal_guidance = kwargs.get("temporal_guidance", False)
        else:
            # When not using score guidance, sample from Gaussian.
            self.sample_start = kwargs.get("sample_start", cfg.DIFFUSION.TRAIN.sample_start)
            self.ddim_step_size = kwargs.get("ddim_step_size", cfg.DIFFUSION.TRAIN.ddim_step_size)

        self.smplh_layer = copy.deepcopy(smplh.layer['neutral'])
        # self.mean_std_dict = pickle.load(open('data/img_feats/diffusion_dataset_stats.pkl', 'rb'))    
    
    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def normalize_betas(self, betas):
        """Normalize SMPL betas to [-1, 1]."""
        return 2 * (betas - self.betas_min) / (self.betas_max - self.betas_min) - 1


    def unnormalize_betas(self, betas):
        """Get back the unnormalized SMPL betas."""
        return 0.5 * (self.betas_max - self.betas_min) * (betas + 1) + self.betas_min


    def load_weights(self, model_dict, strict=False):
        self.model.load_state_dict(model_dict, strict=strict)
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t_: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the added noise, return the denoised parameters x_0.
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters (B: batch_        if self.keypoint_guidance:
            # Ignore hips when fitting to 2D keypoints.
            if not self.use_hips:
                batch["joints_conf"][:, [8, 9, 12, 25 + 2, 25 + 3, 25 + 14]] *= 0.0

            # Ignore GT joints (the first 25 joints are from OpenPose, while the rest ones are GT joints, if they exist).
            batch["joints_conf"][:, 25:] = 0.0

            # Set up camera translation optimizer.
            size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            noise: Tensor of shape [B, P] containing the added noise.
        Returns:
            torch.Tensor of shape [B, P] with the denoised SMPL parameters.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t_: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the clean SMPL parameters x_0, return the added noise.
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            x_0: Tensor of shape [B, P] with the clean SMPL parameters.
        Returns:
            torch.Tensor of shape [B, P] with added noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t - x_0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape)

    def q_sample(self, x_start: torch.Tensor, t_: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Implements q(x_t | x_0).
        Args:
            x_start: Tensor of shape [B, P] with the clean SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            noise: Tensor of shape [B, P] containing the added noise.
        Returns:
            torch.Tensor: Tensor of shape [B, P] containing the noised SMPL parameters.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t_, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape) * noise
        )

    def forward(self, batch: Dict, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = batch["inputs"]["obj_verts_orig"].size(0)
        t_ = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        simple_loss, pred_x_start = self.p_losses(batch, t_, *args, **kwargs)
        return simple_loss, pred_x_start, t_

    def p_losses(self, batch: Dict, t_: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each sample in the batch performs the following:
        - computes noised input x_t at timesteps t.
        - feeds x_t to the denoising model to predict the added noise.
        - computes the training loss.
        Args:
            batch: dictionary containing Tensors with the SMPL parameters and image features.
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
        Reurns:
            torch.Tensor of shape [B, P] with the loss.
            torch.Tensor of shape [B, P] with the predicted clean sample pred_x_start.
        """

        batch_size = batch["inputs"]["obj_verts_orig"].size(0)
        global_orient_aa = batch["targets"]["smpl_pose"][:, :1]
        body_pose_aa = batch["targets"]["smpl_pose"][:, 1:]
        obj_pose_aa = batch['targets']['obj_pose']
        obj_trans = batch['targets']['obj_trans']

        # Get the 6D pose.
        global_orient_6d = rotmat_to_6d(global_orient_aa)
        body_pose_6d = rotmat_to_6d(body_pose_aa)
        obj_pose_6d = rotmat_to_6d(obj_pose_aa)

        pose_6d = torch.cat((global_orient_6d, body_pose_6d), dim=1)
        pose = pose_6d.reshape(batch_size, -1)  # (bs, 52*6)
        # obj = torch.cat((obj_pose_6d.reshape(batch_size, -1), obj_trans), dim=-1)  # (bs, 6 + 3)
        input_data = torch.cat((pose, obj_pose_6d.reshape(batch_size, -1), obj_trans), dim=-1)  # (bs, 52*6 + 6 + 3)
        if self.use_betas:
            # Normalize betas to [-1, 1] and concatenate them with the SMPL pose parameters.
            scaled_betas = normalize_betas(batch["targets"]["smpl_shape"], self.betas_min, self.betas_max)
            params = torch.cat((input_data, scaled_betas), dim=1).float()
        else:
            params = input_data.float()

        # Generate the input noise.
        noise = default(noise, lambda: torch.randn_like(params))
        if "img_feats" in batch["inputs"]:
            cond_img_feats = batch["inputs"]["img_feats"]
        
        if "obj_feat" in batch["inputs"] and self.cfg.DIFFUSION.use_obj_feat:
            cond_geo_feats = batch["inputs"]["obj_feat"].float()
        else:
            cond_geo_feats = batch["inputs"]["obj_verts_orig"].float()

        y_input = {
            "cond_img_feats" : cond_img_feats if "img_feats" in batch["inputs"] else None,
            "cond_geo_feats" : cond_geo_feats,
        }
        # Forward: SAMPLE q(x_t | x_0).
        noised_params = self.q_sample(x_start=params, t_=t_, noise=noise)
        # Reverse: Predict the added input noise.
        pred_noise, pred_x_start = self.model_predictions(noised_params, t_, y = y_input, batch=batch)

        if self.objective == "pred_noise":
            target = noise
            model_output = pred_noise
        elif self.objective == "pred_x0":
            target = params
            model_output = pred_x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        simple_loss = self.loss_fn(model_output, target, reduction="none")
        simple_loss = reduce(simple_loss, "b ... -> b (...)", "mean")
        simple_loss = simple_loss * extract(self.p2_loss_weight, t_, simple_loss.shape)  # (bs, 144)

        return simple_loss, pred_x_start


    def model_predictions(
        self,
        x: torch.Tensor,
        t_: torch.Tensor,
        y: Dict = None,
        batch: Dict = None,
        clip_x_start: bool = True,
        inverse: bool = False,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a noised sample, return the predicted noise (by the denoising model) and the one-step denoised result.
        Args:
            x: Tensor of shape [B, P] containing the noised paramaters x_t (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timesteps (noise levels).
            cond_feats: Tensor of shape [B, C] containing the image features.
            batch: Dictionary containing the dataset and extra information.
            clip_x_start: If True, clip x_start to [-1, 1].
            inverse: If True, we are performing DDIM inversion, so do not compute the conditional score.
        Returns:
            pred_noise: Tensor of shape [B, P] with the predicted noise
            x_start: Tensor of shape [B, P] with the one-step denoised result.
        """
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        model_output = self.model(x, t_, y)
        
        # import pdb; pdb.set_trace()
        mask = None
        if self.objective == "pred_noise":
            pred_noise = model_output

            # TEST-TIME: Compute score guidance and the modified noise prediction.
            if self.use_guidance and not inverse and inference:
                timestep = t_[0]
                if timestep <= self.sample_start:
                    # Compute score guidance, Eq. (8) in the paper.
                    scaled_gradient, mask = self.cond_fn(x, t_, pred_noise, batch)
                    # Compute modified noise prediction, Eq. (10) in the paper.
                    pred_noise += self.sqrt_one_minus_alphas_cumprod[timestep].item() * scaled_gradient

            # One-step denoised result.
            x_start = self.predict_start_from_noise(x, t_, pred_noise)
            x_start = maybe_clip(x_start)

            # TEST-TIME: If using early stopping, cache the samples that converged in the current step.
            if inference and self.use_guidance and self.early_stop and not inverse and mask is not None and torch.any(mask).item():
                self.early_stop_obj.cache_results(mask, x_start, self.camera_translation)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_, x_start)
        return pred_noise, x_start


    def cond_fn(
        self, x: torch.Tensor, t_: torch.Tensor, pred_noise: torch.Tensor, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Functions to compute score guidance.
        Args:
            x : Tensor of shape [B, P] containing the noised paramaters x_t (B: batch_size, P: dimension of SMPL parameters).
            t_ : Tensor of shape [B] containing the current timestep (noise level).
            pred_noise: Tensor of shape [B, P] containing the predicted noise by the denoising model.
            batch : dictionary containing the regression estimates and optionally information for model fitting.
        Returns:
            - The conditional score
            - A mask suggesting which samples coverged in the current step (if applying early stopping).
        """
        # real_batch_size = batch["keypoints_2d"].size(0)
        # num_samples = x.size(0) // real_batch_size

        loss = 0
        mask = None
        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            # Get one-step denoised result with the predicted noise by the denoising model.
            x_start = self.predict_start_from_noise(x_t, t_, pred_noise)
            bs_times_samples = x_start.size(0)
            # pred_pose_6d = x_start[:, :-10] if self.use_betas else x_start
            pred_pose_6d = x_start[:, : 52 * 6]
            pred_obj_pose_6d = x_start[:, 52 * 6: 52 * 6 + 6]
            pred_obj_pose_trans = x_start[:, 52 * 6 + 6: 52 * 6 + 9]
            pred_shape = x_start[:, 52 * 6 + 9:] if self.use_betas else None

            human_pose_ = rot6d_to_aa(pred_pose_6d.reshape(-1,6)).reshape(bs_times_samples,-1)
            human_shape_ = self.unnormalize_betas(pred_shape).float() if self.use_betas else None
            outputs = self.smplh_layer(betas=human_shape_, global_orient=human_pose_[:, :3], body_pose=human_pose_[:, 3:66], left_hand_pose=human_pose_[:, 66:111], right_hand_pose=human_pose_[:, 111:])
            human_verts, human_joints = outputs.vertices - outputs.joints[:, [smplh.root_joint_idx]], outputs.joints - outputs.joints[:, [smplh.root_joint_idx]]
            human_verts = smplh.downsample(human_verts)

            obj_verts = batch["inputs"]["obj_verts_orig"]
            obj_pose = rot6d_to_rotmat(pred_obj_pose_6d).reshape(bs_times_samples, 3, 3)
            obj_verts = torch.matmul(torch.tensor(obj_verts).to(obj_pose.device), obj_pose.transpose(1,2)) + pred_obj_pose_trans[:,None]# - outputs.joints[:, [smplh.root_joint_idx]]

            if self.keypoint_guidance:

                pred_keypoints_3d = human_joints

                # Compute the keypoint reprojection error.
                loss_kp = keypoint_fitting_loss(
                    model_joints=pred_keypoints_3d,
                    camera_translation=self.camera_translation,
                    joints_2d=batch["inputs"]["h2d_keypoints"][:, :, :2],
                    # joints_conf=batch["inputs"]["h2d_keypoints"][:, :, [2]],
                    joints_conf = torch.ones_like(batch["inputs"]["h2d_keypoints"][:, :, [0]]),
                    camera_center=torch.tensor(self.cfg.CAMERA.princpt, device=self.device).repeat(bs_times_samples, 1),
                    focal_length=torch.tensor(self.cfg.CAMERA.focal, device=self.device).repeat(bs_times_samples, 1),
                    img_size=torch.tensor(self.cfg.MODEL.input_img_shape, device=self.device).repeat(bs_times_samples, 1),                )
                ## Update camera translation.
                self.camera_translation_optimizer.zero_grad()
                loss_kp.sum().backward(retain_graph=True)
                self.camera_translation_optimizer.step()

                loss_kp = self.cfg.DIFFUSION.GUIDANCE.w_kp2d * loss_kp
            else:
                loss_kp = torch.tensor(0, device=self.device)

            if self.omask_guidance:
                loss_omask = object_mask_loss(obj_verts, 
                                                batch["meta_info"]["obj_mask"], 
                                                camera_translation=self.camera_translation,
                                                focal_length=torch.tensor(self.cfg.CAMERA.focal, device=self.device).repeat(bs_times_samples, 1),
                                                camera_center=torch.tensor(self.cfg.CAMERA.princpt, device=self.device).repeat(bs_times_samples, 1))
                loss_omask = self.cfg.DIFFUSION.GUIDANCE.w_omask * loss_omask
            else:
                loss_omask = torch.tensor(0, device=self.device)
            

            contact_h = batch["inputs"]["pred_h_contacts"]
            contact_o = batch["inputs"]["pred_o_contacts"]
            if 'pred_o_f_contacts' in batch["inputs"]:
                contact_o_f = batch["inputs"]["pred_o_f_contacts"]
            else:
                contact_o_f = None
            loss_inter, loss_colli, loss_inter_o_f = interaction_loss(obj_verts, human_verts, contact_h,contact_o=contact_o, contact_o_f=contact_o_f,penetration=self.penetration_guidance)
            loss_inter = self.cfg.DIFFUSION.GUIDANCE.w_inter * loss_inter
            loss_colli = self.cfg.DIFFUSION.GUIDANCE.w_colli * loss_colli
            loss_inter_o_f = self.cfg.DIFFUSION.GUIDANCE.w_inter_o_f * loss_inter_o_f
            loss = loss_kp + loss_inter + loss_omask + loss_colli + loss_inter_o_f
            if self.early_stop:
                mask = self.early_stop_obj.get_stop_mask(loss)
                loss[mask] = 0.
            gradient = torch.autograd.grad(loss.sum(), x_t)[0]
            return self.grad_scale * gradient, mask


    @torch.no_grad()
    def sample(self, batch: Dict, cond_img_feats: torch.Tensor, cond_geo_feats: torch.Tensor, batch_size: int, use_guidance=True):
        """
        Run vanilla DDIM or DDIM with score guidance.
        Args:
            batch : Dictionary containing the dataset, corresponding labels and regression predicitons.
            cond_feats: Tensor of shape [B*N, C] containing the images features. (B: batch_size, N: number of samples to draw for each image in the batch, C: dimension of image feature).
            batch_size: B*N
        """
        shape = (batch_size, self.model.diffusion_dim + 10)
        sample_fn = self.ddim_with_guidance if use_guidance else self.ddim_vanilla
        return sample_fn(batch, cond_img_feats, cond_geo_feats, shape)


    @torch.no_grad()
    def ddim_vanilla(
            self,
            batch: Dict,
            cond_img_feats: torch.Tensor,
            cond_geo_feats: torch.Tensor,
            shape: Tuple,
            clip_denoised: bool = True,
            eta: float = 0.0
        ) -> Dict:
        """
        Vanilla DDIM sampling.
        Args:
            batch: Dictionary with the dataset and corresponding labels.
            cond_feats: Tensor of shape [B*N, C] containing the image feautres.
            shape : Tuple with the shape of the parameters used in the diffusion model.
            clip_denoised: Flag that indicates whether the denoised result should be scaled to [-1, 1].
            eta: DDIM eta parameter, eta=0 corresponding to deterministic DDIM sampling.
        Returns:
            Dictionary with x_start.
        """
        batch_size = shape[0]
        times = list(range(0, self.cfg.DIFFUSION.TRAIN.sample_start + 1, self.cfg.DIFFUSION.TRAIN.ddim_step_size))
        times_next = [-1] + list(times[:-1])
        time_pairs = list(
            reversed(list(zip(times[1:], times_next[1:])))
        )  # [ ..., (20, 10), (10, 0)]

        x_t = torch.randn(shape, device=self.device)
        y_input = {
            "cond_img_feats" : cond_img_feats,
            "cond_geo_feats" : cond_geo_feats,
        }
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t,
                time_cond,
                y = y_input,
                batch=batch,
                clip_x_start=clip_denoised,
            )
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return {'x_0': x_start,
                'camera_translation': None}


    @torch.no_grad()
    def ddim_with_guidance(
        self,
        batch: Dict,
        cond_img_feats: torch.Tensor,
        cond_geo_feats: torch.Tensor,
        shape: Tuple,
        clip_denoised: bool = False,
        eta: float = 0.0,
    ) -> Dict:
        """
        DDIM sampling with guidance.
        Args:
            batch: Dictionary with the dataset, labels and regression predictions.
            cond_feats: Tensor of shape [B*N, C] with the image feautres.
            shape : Tuple with the shape of the parameters used in the diffusion model.
            clip_denoised: Flag that indicates whether the denoised result should be scaled to [-1, 1].
            eta: DDIM eta parameter, eta=0 corresponding to deterministic DDIM sampling.
        Returns:
            Dictionary with the refined x_start and optionally the optimized camera_translation.
        """

        batch_size = shape[0]
        times = list(range(0, self.sample_start + 1, self.ddim_step_size))
        times_next = [-1] + list(times[:-1])
        time_pairs = list(
            reversed(list(zip(times[1:], times_next[1:])))
        )  # [ ..., (20, 10), (10, 0)]
        time_pairs_inv = list(
            zip(times_next[1:], times[1:])
        )  # [(0, 10), (10, 20), ... ]

        self.camera_translation = None
        if self.keypoint_guidance:
            # Ignore hips when fitting to 2D keypoints.
            if not self.use_hips:
                batch["inputs"]["h2d_keypoints"][:,[8, 9, 12, 25 + 2, 25 + 3, 25 + 14], [2]]*= 0.0

            # Ignore GT joints (the first 25 joints are from OpenPose, while the rest ones are GT joints, if they exist).
            if batch["inputs"]["h2d_keypoints"].shape[2]>2:
                batch["inputs"]["h2d_keypoints"][:, 25:, [2]] = 0.0

            # Set up camera translation optimizer.
            self.camera_translation = copy.deepcopy(batch["inputs"]["pred_cam_trans"])
            self.camera_translation.requires_grad_(True)
            self.camera_translation_optimizer = torch.optim.Adam([self.camera_translation], lr=1e-1)


        # Initialize object for early stopping.
        if self.early_stop:
            self.early_stop_obj = EarlyStopping(shape=shape, device=self.device, opt_trans=True if self.keypoint_guidance else False)

        x_start, x_t, noise = self.q_sample_verbose(batch, shape, timestep=time_pairs_inv[0][0])
        x_t = x_start.clone()

        y_input = {
            "cond_img_feats" : cond_img_feats,
            "cond_geo_feats" : cond_geo_feats,
        }

        for _ in range(self.optim_iters):

            ### DDIM Inversion ###
            for time, time_next in time_pairs_inv:
                time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
                pred_noise, x_start = self.model_predictions(
                    x_t,
                    time_cond,
                    y = y_input,
                    batch=batch,
                    clip_x_start=clip_denoised,
                    inverse=True,
                )
                alpha_next = self.alphas_cumprod[time_next]
                x_t = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

            ### DDIM Sample Loop ###
            for time, time_next in time_pairs:
                time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
                pred_noise, x_start = self.model_predictions(
                    x_t,
                    time_cond,
                    y = y_input,
                    batch=batch,
                    clip_x_start=clip_denoised,
                    inference=True
                )
                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma**2).sqrt()
                # 使用当前时间戳作为随机种子
                torch.manual_seed(int(time.time() * 1000000))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(int(time.time() * 1000000))
                noise = torch.randn_like(x_t)
                x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # Return the output.
        if self.camera_translation is not None:
            self.camera_translation = self.camera_translation.detach()
        output = (
                    self.early_stop_obj.get_preds(x_start, self.camera_translation)
                    if self.early_stop
                    else {'x_0': x_start, 'camera_translation': self.camera_translation}
        )
        return output

    @torch.no_grad()
    def one_step_ddim_with_guidance(
        self,
        batch: Dict,
        x_t,
        y_input,
        time_pairs_inv,
        time_pairs,
        shape: Tuple,
        clip_denoised: bool = False,
        eta: float = 0.0,
    ) -> Dict:
        """
        DDIM sampling with guidance.
        Args:
            batch: Dictionary with the dataset, labels and regression predictions.
            cond_feats: Tensor of shape [B*N, C] with the image feautres.
            shape : Tuple with the shape of the parameters used in the diffusion model.
            clip_denoised: Flag that indicates whether the denoised result should be scaled to [-1, 1].
            eta: DDIM eta parameter, eta=0 corresponding to deterministic DDIM sampling.
        Returns:
            Dictionary with the refined x_start and optionally the optimized camera_translation.
        """

        batch_size = shape[0]

        ### DDIM Inversion ###
        for time_step, time_next in time_pairs_inv:
            time_cond = torch.full((batch_size,), time_step, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t,
                time_cond,
                y = y_input,
                batch=batch,
                clip_x_start=clip_denoised,
                inverse=True,
            )
            alpha_next = self.alphas_cumprod[time_next]
            x_t = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

        ### DDIM Sample Loop ###
        for time_step, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time_step, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t,
                time_cond,
                y = y_input,
                batch=batch,
                clip_x_start=clip_denoised,
                inference=True
            )
            alpha = self.alphas_cumprod[time_step]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            time_stamp = int(time.time() * 1000000)
            torch.cuda.manual_seed(time_stamp)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed(int(time.time() * 1000000))
            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # Return the output.
        if self.camera_translation is not None:
            self.camera_translation = self.camera_translation.detach()
        # output = (
        #             self.early_stop_obj.get_preds(x_start, self.camera_translation)
        #             if self.early_stop
        #             else {'x_0': x_start, 'camera_translation': self.camera_translation}
        # )
        output = {
            "x_0": x_start,
            "camera_translation": self.camera_translation,
            "x_t": x_t,
            "noise": noise
        }
        return output

    @torch.no_grad()
    def one_step_ddim_without_guidance(
        self,
        batch: Dict,
        x_t,
        y_input,
        time_pairs_inv,
        time_pairs,
        shape: Tuple,
        clip_denoised: bool = False,
        eta: float = 0.0,
    ) -> Dict:
        """
        DDIM sampling with guidance.
        Args:
            batch: Dictionary with the dataset, labels and regression predictions.
            cond_feats: Tensor of shape [B*N, C] with the image feautres.
            shape : Tuple with the shape of the parameters used in the diffusion model.
            clip_denoised: Flag that indicates whether the denoised result should be scaled to [-1, 1].
            eta: DDIM eta parameter, eta=0 corresponding to deterministic DDIM sampling.
        Returns:
            Dictionary with the refined x_start and optionally the optimized camera_translation.
        """

        batch_size = shape[0]

        ### DDIM Inversion ###
        for time_step, time_next in time_pairs_inv:
            time_cond = torch.full((batch_size,), time_step, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t,
                time_cond,
                y = y_input,
                batch=batch,
                clip_x_start=clip_denoised,
                inverse=True,
            )
            alpha_next = self.alphas_cumprod[time_next]
            x_t = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

        ### DDIM Sample Loop ###
        for time_step, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time_step, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(
                x_t,
                time_cond,
                y = y_input,
                batch=batch,
                clip_x_start=clip_denoised,
                inference=False
            )
            alpha = self.alphas_cumprod[time_step]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            time_stamp = int(time.time() * 1000000)
            torch.cuda.manual_seed(time_stamp)
            # if torch.cuda.is_available():
            #     torch.cuda.manual_seed(int(time.time() * 1000000))
            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # Return the output.
        if self.camera_translation is not None:
            self.camera_translation = self.camera_translation.detach()
        output = {
            "x_0": x_start,
            "camera_translation": self.camera_translation,
            "x_t": x_t,
            "noise": noise
        }
        return output

    def q_sample_verbose(self, batch: Dict, shape: Tuple, timestep: int = 999) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Similar to q_sample(), but also returns the x_start and added noise.
        It also converts the SMPL from a regression model to the appropriate format used by the Diffusion Model.
        Args:
            batch : Dictionary containing SMPL parameter predtictions.
            shape : Tuple containing the shape of the diffusion model parameters.
            timestep : Timestep (noise level) to use.
        Returns:
            x_start : Tensor of shape [B*N, P] containing the input (of the diffusion model) for the regression estimate.
            x_t : Tensor of shape [B*N, P] containing the noised input at noise level t.
            noise: Tensor of shape [B*N, P] containing the the added noise to x_start to produce x_t.
        """
        batch_size = shape[0]
        pred_pose_6d = batch["inputs"]["pred_pose"]  # bs, 52 * 6
        true_batch_size = pred_pose_6d.size(0)
        pred_pose_6d = pred_pose_6d.reshape(true_batch_size, -1)
        pred_obj_pose_6d = batch["inputs"]["pred_obj_pose"].reshape(true_batch_size, -1)    # bs, 6
        pred_obj_trans = batch["inputs"]["pred_obj_trans"]  #bs, 3
        pred_pose_6d = torch.cat((pred_pose_6d, pred_obj_pose_6d, pred_obj_trans), dim=-1)
        # pred_pose_6d = pred_pose_rotmat[:, :, :, :2].permute(0, 1, 3, 2)
        num_samples = batch_size // true_batch_size

        x_start = pred_pose_6d.reshape(true_batch_size, -1)
        # Potentially include SMPL betas in x_start.
        if self.use_betas:
            scaled_betas = normalize_betas(batch["inputs"]["pred_betas"], self.betas_min, self.betas_max)
            x_start = torch.cat((x_start, scaled_betas), dim=1).float()
        x_start = x_start.unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size, -1)

        time_cond = torch.full( (batch_size,), timestep, device=self.device, dtype=torch.long)
        # 0-th sample is mode, and if num_samples > 1 we also sample noise
        noise_mode = torch.zeros((true_batch_size, 1, x_start.size(-1)), device=self.device)
        noise_samples = torch.randn((true_batch_size, num_samples - 1, x_start.size(-1)), device=self.device)
        noise = torch.cat((noise_mode, noise_samples), dim=1).reshape(true_batch_size * num_samples, -1)

        x_t = self.q_sample(x_start=x_start, t_=time_cond, noise=noise)

        return x_start, x_t, noise
    
    def prepare_xt(
        self,
        batch: Dict,
        cond_img_feats: torch.Tensor,
        cond_geo_feats: torch.Tensor,
        shape: Tuple,
        force_sample_start=-1
    ):
        batch_size = shape[0]
        if force_sample_start > 0:
            sample_start = force_sample_start
        else:
            sample_start = self.sample_start
        times = list(range(0, sample_start + 1, self.ddim_step_size))
        times_next = [-1] + list(times[:-1])
        time_pairs = list(
            reversed(list(zip(times[1:], times_next[1:])))
        )  # [ ..., (20, 10), (10, 0)]
        time_pairs_inv = list(
            zip(times_next[1:], times[1:])
        )  # [(0, 10), (10, 20), ... ]

        self.camera_translation = None
        if self.keypoint_guidance:
            # Ignore hips when fitting to 2D keypoints.
            if not self.use_hips:
                batch["inputs"]["h2d_keypoints"][:,[8, 9, 12, 25 + 2, 25 + 3, 25 + 14], [2]]*= 0.0

            # Ignore GT joints (the first 25 joints are from OpenPose, while the rest ones are GT joints, if they exist).
            if batch["inputs"]["h2d_keypoints"].shape[2]>2:
                batch["inputs"]["h2d_keypoints"][:, 25:, [2]] = 0.0

            # Set up camera translation optimizer.
            self.camera_translation = copy.deepcopy(batch["inputs"]["pred_cam_trans"])
            self.camera_translation.requires_grad_(True)
            self.camera_translation_optimizer = torch.optim.Adam([self.camera_translation], lr=1e-1)


        # Initialize object for early stopping.
        if self.early_stop:
            self.early_stop_obj = EarlyStopping(shape=shape, device=self.device, opt_trans=True if self.keypoint_guidance else False)

        x_start, x_t, noise = self.q_sample_verbose(batch, shape, timestep=time_pairs_inv[0][0])
        x_t = x_start.clone()

        y_input = {
            "cond_img_feats" : cond_img_feats,
            "cond_geo_feats" : cond_geo_feats,
        }
        return x_t, y_input, time_pairs_inv, time_pairs, shape