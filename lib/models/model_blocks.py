import math
import torch
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_emb_dim=None):
        super().__init__()

        inp_dim = input_dim
        output_dim = input_dim
        if cond_emb_dim is not None:
            inp_dim += cond_emb_dim

        self.layer_norm1 = nn.LayerNorm(inp_dim)
        self.linear_layer1 = nn.Linear(inp_dim, hidden_dim)
        self.act1 = nn.GELU()

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.linear_layer1(self.layer_norm1(x)))
        x = self.act2(self.linear_layer2(self.layer_norm2(x)))
        return x


class ResMLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, time_emb_dim=None, cond_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, input_dim * 2))
            if time_emb_dim is not None
            else None
        )

        self.block = MLPBlock(input_dim, hidden_dim, cond_emb_dim)

    def forward(self, x, time_emb=None, cond_emb=None):
        # Scale and shift based on time embedding.
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        residual = x

        # Concatenate params and image features.
        if cond_emb is not None:
            x = torch.cat((x, cond_emb), dim=1)

        x = residual + self.block(x)

        return x

class AttentionCondEmb(nn.Module):
    def __init__(self, input_dim, cond_emb_dim, num_heads=1):
        super().__init__()
        # 将条件嵌入映射到输入维度空间
        self.query_proj = nn.Linear(cond_emb_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
        # 使用简单的多头注意力
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        
    def forward(self, x, cond_emb):
        if x.dim() == 2 and cond_emb.dim() == 2:
            # 获取条件嵌入的查询
            q = self.query_proj(cond_emb).unsqueeze(0)  # (1, batch_size, cond_emb_dim)
            
            # 对输入特征进行投影
            k = self.key_proj(x).unsqueeze(0)  # (1, batch_size, input_dim)
            v = self.value_proj(x).unsqueeze(0)  # (1, batch_size, input_dim)
        elif x.dim() == 3 and cond_emb.dim() == 3:
            q = self.query_proj(cond_emb)
            k = self.key_proj(x)
            v = self.value_proj(x)
        else:
            raise ValueError("Input and condition embedding must have the same number of dimensions. [2 or 3]")

        # 进行注意力计算
        attn_output, _ = self.attn(q, k, v)  # (1, batch_size, input_dim)
        
        # 结果与原始输入特征相加
        return x + attn_output.squeeze(0)

class AttnMLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_emb_dim=None, is_ip=False):
        super().__init__()

        inp_dim = input_dim
        output_dim = input_dim

        self.layer_norm1 = nn.LayerNorm(inp_dim)
        self.linear_layer1 = nn.Linear(inp_dim, hidden_dim)
        self.act1 = nn.GELU()

        self.attn_cond_emb = AttentionCondEmb(hidden_dim, cond_emb_dim=cond_emb_dim)
        if is_ip:
            self.ip_attn_cond_emb = AttentionCondEmb(hidden_dim, cond_emb_dim=cond_emb_dim)
            self.after_ip_attn = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.GELU()

    def forward(self, x, cond_emb=None, ip_cond_emb=None, ip_scale=1.0):
        x = self.act1(self.linear_layer1(self.layer_norm1(x)))
        if ip_cond_emb is None:
            x = self.attn_cond_emb(x, cond_emb)
        else:
            x = self.attn_cond_emb(x, cond_emb) + self.ip_attn_cond_emb(x, ip_cond_emb) * ip_scale
            x = self.after_ip_attn(x)
        x = self.act2(self.linear_layer2(self.layer_norm2(x)))
        return x

class AttnResMLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, time_emb_dim=None, cond_emb_dim=None, is_ip=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, input_dim * 2))
            if time_emb_dim is not None
            else None
        )

        self.is_ip = is_ip
        if is_ip:
            self.block = AttnMLPBlock(input_dim, hidden_dim, cond_emb_dim, is_ip=True)
            self.align_ip_cond = nn.Linear(cond_emb_dim, cond_emb_dim)

        else:
            self.block = AttnMLPBlock(input_dim, hidden_dim, cond_emb_dim)

    def forward(self, x, time_emb=None, cond_emb=None, ip_cond_emb=None, ip_scale=1.0):
        # Scale and shift based on time embedding.
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        residual = x    # b, 321

        if ip_cond_emb is None and not self.is_ip:
            x = residual + self.block(x, cond_emb)
        else:
            ip_cond_emb = self.align_ip_cond(ip_cond_emb)
            x = residual + self.block(x, cond_emb, ip_cond_emb, ip_scale)

        return x