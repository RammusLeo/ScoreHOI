import torch
from torch import nn
from .model_blocks import SinusoidalPosEmb, ResMLPBlock


PREDICTORS = {
    "prohmr": {"thetas_emb_dim": 2048 + 1024, "betas_emb_dim": 2048},
    "pare": {"thetas_emb_dim": 3072, "betas_emb_dim": 1536},
}

'''
IMG_FEATS: pare
POSE_DIM : 144
SHAPE_DIM : 10
NUM_BLOCKS_POSE: 3
NUM_BLOCKS_SHAPE: 2
HIDDEN_LAYER_DIM: 1024
OBJECTIVE: pred_noise
LOSS_TYPE: l2
'''

class FC(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.DIFFUSION.use_betas
        img_feats = 'prohmr'
        hidden_dim = 1024
        # diffusion dimensions
        self.diffusion_dim = 52 * 6 + 6 + 3
        self.thetas_obj_dim = 52 * 6 + 6 +3
        self.betas_dim = 10 if self.use_betas else 0
        # image features
        self.thetas_emb_dim = PREDICTORS[img_feats]["thetas_emb_dim"]
        self.betas_emb_dim = PREDICTORS[img_feats]["betas_emb_dim"]

        # SMPL thetas
        time_dim = self.thetas_obj_dim * 4
        sinu_pos_emb = SinusoidalPosEmb(self.thetas_obj_dim)
        fourier_dim = self.thetas_obj_dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_mlp = nn.Linear(in_features=self.thetas_obj_dim, out_features=self.thetas_obj_dim)
        self.blocks = nn.ModuleList([])
        for _ in range(cfg.DIFFUSION.DENOISING_MODEL.num_block_pose):
            self.blocks.append(
                ResMLPBlock(
                    input_dim=self.thetas_obj_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim,
                    cond_emb_dim=self.thetas_emb_dim,
                )
            )
        self.final_mlp = nn.Linear(in_features=self.thetas_obj_dim, out_features=self.thetas_obj_dim)

        # SMPL betas (optionally)
        if self.use_betas:
            time_dim = self.betas_dim * 4
            sinu_pos_emb = SinusoidalPosEmb(self.betas_dim)
            fourier_dim = self.betas_dim
            self.time_mlp_betas = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.init_mlp_betas = nn.Linear(in_features=self.betas_dim, out_features=self.betas_dim)
            self.blocks_betas = nn.ModuleList([])
            for _ in range(cfg.DIFFUSION.DENOISING_MODEL.num_block_shape):
                self.blocks_betas.append(
                    ResMLPBlock(
                        input_dim=self.betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim,
                        cond_emb_dim=self.betas_emb_dim,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=self.betas_dim, out_features=self.betas_dim)
        self.geo_cond_embedding = nn.Linear(64*3, 1024)


    def forward(self, x: torch.Tensor, time: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            time : Tensor of shape [B] containing timesteps.
            cond_emb : Tensor of shape [B, cond_emb_dim] containing the image features to condition the model.
        Returns:
            torch.Tensor : predicted noise with shape [B, P].
        """
        # if self.use_betas:
        #     thetas = x[:, :-10]
        #     betas = x[:, -10:]
        #     if self.split_img_emb:
        #         thetas_emb = cond_emb[:, :3072]
        #         cam_shape_emb = cond_emb[:, 3072:]
        # else:
        thetas = x
        bs = x.shape[0]
        cond_img_feat = y["cond_img_feats"]   # B，2048
        cond_geo_feat = y["cond_geo_feats"].reshape(bs, -1)     #B, 64 * 3
        cond_geo_feat = self.geo_cond_embedding(cond_geo_feat)  # B, 1024
        cond_emb = torch.cat((cond_img_feat, cond_geo_feat), dim=-1) #B, 3072


        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)
        for block in self.blocks:
            thetas = block(thetas, tt, cond_emb)
        thetas = self.final_mlp(thetas)
        return thetas
        # if self.use_betas:
        #     betas = self.init_mlp_betas(betas)
        #     tt_betas = self.time_mlp_betas(time)
        #     for block in self.blocks_betas:
        #         betas = block(
        #             betas, tt_betas, cam_shape_emb if self.split_img_emb else cond_emb
        #         )
        #     betas = self.final_mlp_betas(betas)

        #     thetas_betas = torch.cat((thetas, betas), dim=1)
        #     return thetas_betas
        # else:
        #     return thetas
