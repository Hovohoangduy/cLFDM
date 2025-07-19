import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
from einops import rearrange, repeat
from sync_batchnorm import DataParallelWithCallback
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.vfdm_with_gentron import GaussianDiffusionGenTron
from DM.modules.text import tokenize, bert_embed

# Positional Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

# Transformer Denoiser for 2-channel flow
class DiffusionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, time_emb_dim):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, dim))
        self.to_patch = None
        self.to_out = nn.Conv3d(dim, 2, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, batch_first=True),
                'norm2': nn.LayerNorm(dim),
                'mlp': nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
            }) for _ in range(depth)
        ])
        self.temp_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x, t, cond=None, cond_scale=1.0):
        B, C, F, H, W = x.shape
        if self.to_patch is None:
            self.to_patch = nn.Conv3d(C, self.dim, (1,7,7), padding=(0,3,3)).to(x.device)
        x = self.to_patch(x)
        x = rearrange(x, 'b d f h w -> (b f) (h w) d')
        t_emb = repeat(self.time_mlp(t), 'b d -> (b f) n d', f=F, n=H*W)
        x = x + t_emb
        for blk in self.blocks:
            h_ = blk['norm1'](x)
            att, _ = blk['attn'](h_, h_, h_)
            x = x + att + blk['mlp'](blk['norm2'](x))
        x_t, _ = self.temp_attn(
            rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F),
            rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F),
            rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F)
        )
        x = rearrange(x_t, '(n b) f d -> (b f) n d', b=B, f=F)
        x = rearrange(x, '(b f) (h w) d -> b d f h w', b=B, f=F, h=H, w=W)
        return self.to_out(x)

# Main FlowDiffusion with GenTron for multi-GPU
class FlowDiffusionGenTron(nn.Module):
    def __init__(
        self,
        img_size=32,
        num_frames=40,
        sampling_timesteps=250,
        null_cond_prob=0.1,
        ddim_sampling_eta=1.,
        timesteps=1000,
        dim=64,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=256,
        lr=1e-4,
        adam_betas=(0.9,0.999),
        is_train=True,
        use_residual_flow=False,
        pretrained_pth="",
        config_pth=""
    ):
        super().__init__()
        self.use_residual_flow = use_residual_flow
        # load config & checkpoint
        cfg = yaml.safe_load(open(config_pth))
        ckpt = torch.load(pretrained_pth) if pretrained_pth else None
        # spatial modules
        self.generator = Generator(**cfg['model_params']['generator_params']).cuda()
        self.region_predictor = RegionPredictor(**cfg['model_params']['region_predictor_params']).cuda()
        self.bg_predictor = BGMotionPredictor(**cfg['model_params']['bg_predictor_params']).cuda()
        if ckpt:
            self.generator.load_state_dict(ckpt['generator'])
            self.region_predictor.load_state_dict(ckpt['region_predictor'])
            self.bg_predictor.load_state_dict(ckpt['bg_predictor'])
        for net in (self.generator, self.region_predictor, self.bg_predictor): net.eval(); net.requires_grad_(False)
        # diffusion wrapper
        denoiser = DiffusionTransformer(dim, depth, heads, dim_head, mlp_dim, time_emb_dim=dim)
        self.diffusion = GaussianDiffusionGenTron(
            denoise_fn=denoiser,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
            loss_type='l2',
            use_dynamic_thres=True
        ).cuda()
        # optimizer
        if is_train:
            self.optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=adam_betas)
        # placeholders
        self.ref_img = None; self.real_vid = None; self.ref_text = None

    def set_train_input(self, real_vid, ref_img, ref_text):
        self.real_vid = real_vid.cuda()
        self.ref_img = ref_img.cuda()
        self.ref_text = ref_text

    def forward(self, real_vid=None, ref_img=None, ref_text=None):
        if real_vid is not None: self.real_vid = real_vid.cuda()
        if ref_img is not None: self.ref_img = ref_img.cuda()
        if ref_text is not None: self.ref_text = ref_text
        b, c, nf, h, w = self.real_vid.shape
        # compute pseudo flow
        grid, conf = [], []
        with torch.no_grad():
            src = self.region_predictor(self.ref_img)
            for i in range(nf):
                drv = self.real_vid[:,:,i]
                out = self.generator(self.ref_img, src, self.region_predictor(drv), self.bg_predictor(self.ref_img, drv))
                grid.append(out['optical_flow'].permute(0,3,1,2))
                conf.append(out['occlusion_map'])
        flow = torch.stack(grid,2)
        feat = out['bottle_neck_feat'].detach()
        # loss
        loss, null_mask = self.diffusion(
            flow if not self.use_residual_flow else flow - self.get_identity_grid(b,nf,h//4,w//4),
            feat,
            self.ref_text
        )
        self.rec_loss = loss; self.null_cond_mask = null_mask
        return loss

    def optimize_parameters(self):
        l = self.forward(); self.optimizer.zero_grad(); l.backward(); self.optimizer.step(); self.rec_warp_loss = l

    def sample_video(self, sample_img, sample_text, cond_scale=1.0):
        feat = self.generator.compute_fea(sample_img.cuda())
        return self.diffusion.sample(feat, cond=sample_text, cond_scale=cond_scale)

    def get_identity_grid(self, b, nf, H, W):
        h = torch.linspace(-1,1,H,device='cuda'); w = torch.linspace(-1,1,W,device='cuda')
        g = torch.stack(torch.meshgrid(h, w),-1).flip(2)
        return g.unsqueeze(0).unsqueeze(2).repeat(b,1,nf,1,1).permute(0,3,2,1,4)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES","")
    model = FlowDiffusionGenTron(
        img_size=128, num_frames=40, sampling_timesteps=250,
        null_cond_prob=0.1, ddim_sampling_eta=1.0, timesteps=1000,
        dim=64, depth=4, heads=8, dim_head=64, mlp_dim=256,
        lr=1e-4, adam_betas=(0.9,0.999), is_train=True,
        use_residual_flow=False,
        pretrained_pth='pretrained.pth', config_pth='config.yaml'
    )
    model = DataParallelWithCallback(model)
    # dummy
    bs, nf, sz = 8, 40, 128
    real_vid = torch.rand(bs,3,nf,sz,sz).cuda()
    ref_img = torch.rand(bs,3,sz,sz).cuda()
    cond_text = ['demo']*bs
    cond = bert_embed(tokenize(cond_text), return_cls_repr=True).cuda()
    model.module.set_train_input(real_vid, ref_img, cond)
    loss = model(real_vid=real_vid, ref_img=ref_img, ref_text=cond)
    sample = model.module.sample_video(ref_img, ['demo'], cond_scale=1.0)
