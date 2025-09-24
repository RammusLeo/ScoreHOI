import torch
from torch import nn
from .model_blocks import SinusoidalPosEmb, AttnResMLPBlock


class FC(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.DIFFUSION.use_betas
        hidden_dim = 2048
        # diffusion dimensions
        self.diffusion_dim = 52 * 6 + 6 + 3
        self.thetas_obj_dim = 52 * 6 + 6 +3
        self.betas_dim = 10 if self.use_betas else 0
        # image features
        self.thetas_emb_dim = 2048
        self.betas_emb_dim = 2048

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
                AttnResMLPBlock(
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
                    AttnResMLPBlock(
                        input_dim=self.betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim,
                        cond_emb_dim=self.betas_emb_dim,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=self.betas_dim, out_features=self.betas_dim)
        self.geo_cond_embedding = nn.Linear(64*3, 2048)


    def forward(self, x: torch.Tensor, time: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            time : Tensor of shape [B] containing timesteps.
            cond_emb : Tensor of shape [B, cond_emb_dim] containing the image features to condition the model.
        Returns:
            torch.Tensor : predicted noise with shape [B, P].
        """
        thetas = x
        bs = x.shape[0]
        # cond_img_feat = y["cond_img_feats"]   # B，2048
        cond_geo_feat = y["cond_geo_feats"].reshape(bs, -1)     #B, 64 * 3
        cond_geo_feat = self.geo_cond_embedding(cond_geo_feat)  # B, 1024
        # cond_emb = torch.cat((cond_img_feat, cond_geo_feat), dim=-1) #B, 3072
        cond_emb = cond_geo_feat

        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)
        for block in self.blocks:
            thetas = block(thetas, tt, cond_emb)
        thetas = self.final_mlp(thetas)
        return thetas

class IPFC(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.DIFFUSION.use_betas
        hidden_dim = 2048
        # diffusion dimensions
        self.diffusion_dim = 52 * 6 + 6 + 3
        self.thetas_obj_dim = 52 * 6 + 6 +3
        self.betas_dim = 10 if self.use_betas else 0
        # image features
        self.thetas_emb_dim = 2048
        self.betas_emb_dim = 2048

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
                AttnResMLPBlock(
                    input_dim=self.thetas_obj_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim,
                    cond_emb_dim=self.thetas_emb_dim,
                    is_ip=True
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
                    AttnResMLPBlock(
                        input_dim=self.betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim,
                        cond_emb_dim=self.betas_emb_dim,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=self.betas_dim, out_features=self.betas_dim)
        self.geo_cond_embedding = nn.Linear(64*3, 2048)

    def load_weights(self, checkpoint):
        self.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor, time: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            time : Tensor of shape [B] containing timesteps.
            cond_emb : Tensor of shape [B, cond_emb_dim] containing the image features to condition the model.
        Returns:
            torch.Tensor : predicted noise with shape [B, P].
        """

        thetas = x
        bs = x.shape[0]
        cond_img_feat = y["cond_img_feats"]   # B，2048
        cond_geo_feat = y["cond_geo_feats"].reshape(bs, -1)     #B, 64 * 3
        cond_geo_feat = self.geo_cond_embedding(cond_geo_feat)  # B, 1024
        # cond_emb = torch.cat((cond_img_feat, cond_geo_feat), dim=-1) #B, 3072
        cond_emb = cond_geo_feat

        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)
        for block in self.blocks:
            thetas = block(thetas, tt, cond_emb, cond_img_feat)
        thetas = self.final_mlp(thetas)
        return thetas

class OBJFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=2048):
        super(OBJFeatureExtractor, self).__init__()
        
        # 隐藏层1：输入维度为 input_dim，输出维度为较大的 4096
        self.fc1 = nn.Linear(input_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        
        # 隐藏层2：将维度降到 2048
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        
        # 隐藏层3：进一步降到目标维度 2048
        self.fc3 = nn.Linear(2048, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
        # Dropout层，用于正则化
        self.dropout = nn.Dropout(0.3)  # 30%的Dropout

    def forward(self, x):
        # 通过每一层，加入ReLU激活和BatchNorm
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Dropout可以放在激活之后
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # 最后输出2048维的特征
        return x

class IPFC_Obj(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.DIFFUSION.use_betas
        hidden_dim = 2048
        # diffusion dimensions
        self.diffusion_dim = 52 * 6 + 6 + 3
        self.thetas_obj_dim = 52 * 6 + 6 +3
        self.betas_dim = 10 if self.use_betas else 0
        # image features
        self.thetas_emb_dim = 2048
        self.betas_emb_dim = 2048

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
                AttnResMLPBlock(
                    input_dim=self.thetas_obj_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim,
                    cond_emb_dim=self.thetas_emb_dim,
                    is_ip=True
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
                    AttnResMLPBlock(
                        input_dim=self.betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim,
                        cond_emb_dim=self.betas_emb_dim,
                        is_ip=True,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=self.betas_dim, out_features=self.betas_dim)
        self.geo_cond_embedding_feat = OBJFeatureExtractor(64*3+1024, 2048)

    def load_weights(self, checkpoint):
        self.load_state_dict(checkpoint)

    def forward(self, x: torch.Tensor, time: torch.Tensor, y: dict = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            time : Tensor of shape [B] containing timesteps.
            cond_emb : Tensor of shape [B, cond_emb_dim] containing the image features to condition the model.
        Returns:
            torch.Tensor : predicted noise with shape [B, P].
        """
        if self.use_betas:
            thetas = x[:, :-10]
            betas = x[:, -10:]
        else:
            thetas = x
            
        bs = x.shape[0]
        cond_img_feat = y["cond_img_feats"]   # B，2048
        cond_geo_feat = y["cond_geo_feats"].reshape(bs, -1)     #B, 64 * 3
        cond_geo_feat = self.geo_cond_embedding_feat(cond_geo_feat)  # B, 1024
        # cond_emb = torch.cat((cond_img_feat, cond_geo_feat), dim=-1) #B, 3072
        cond_emb = cond_geo_feat
        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)
        for block in self.blocks:
            thetas = block(thetas, tt, cond_emb, cond_img_feat)
        thetas = self.final_mlp(thetas)
        
        if self.use_betas:
            betas = self.init_mlp_betas(betas)
            tt_betas = self.time_mlp_betas(time)
            for block in self.blocks_betas:
                betas = block(
                    betas, tt_betas, cond_emb, cond_img_feat
                )
            betas = self.final_mlp_betas(betas)

            thetas_betas = torch.cat((thetas, betas), dim=1)
            return thetas_betas
        else:
            return thetas