import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim = 384, #small_depth = 4, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 384,  #large_dim_head = 64, #large_depth = 1, large_heads = 3,large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                # nn.Linear(small_dim, large_dim),
                # nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, num_heads = cross_attn_heads, attn_drop = dropout)),
                nn.LayerNorm(large_dim, eps=1e-6),
                
                # nn.Linear(large_dim, small_dim),
                # nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, num_heads = cross_attn_heads, attn_drop = dropout)),
                nn.LayerNorm(small_dim, eps=1e-6),
                
            ]))

    def forward(self, xs, xl):
        # print (len(xs), xs[0].shape)
        # xs = self.transformer_enc_small(xs)
        # xl = self.transformer_enc_large(xl)

        for cross_attn_s, n_l,\
            cross_attn_l, n_s in self.cross_attn_layers:
        # for f_sl, g_ls, cross_attn_s in self.cross_attn_layers:          
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = large_class.unsqueeze(1)
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            # cal_out = cal_out
            xl = torch.cat((cal_out, x_large), dim=1)
            xl = n_l(xl)

            # Cross Attn for Smaller Patch
            cal_q = small_class.unsqueeze(1)
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            # cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)
            xs = n_s(xs)

        return xs, xl



        
   
# vit_b = 768; vit_s = 384    
class Fus_CrossViT(nn.Module):
    def __init__(self, model_vit_cxr, model_vit_enh, num_classes=3, small_dim = 384,
                 large_dim = 384, cross_attn_depth = 1, multi_scale_enc_depth = 1,
                 heads = 3, dropout = 0.,
                 pool = 'cls'):
        super().__init__()
        
        # self.cfg = cfg
        self.vit_features_cxr = model_vit_cxr.features3D
        # self.vit_cxr = model_vit_cxr
        
        self.vit_features_enh = model_vit_enh.features3D
        # self.vit_enh = model_vit_enh
        
        self.multi_scale_transformers = nn.ModuleList([])
        
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim,
                                                                               # small_dim_head=small_dim//heads,
                                                                              # small_mlp_dim=small_dim*scale_dim,
                                                                              
                                                                              large_dim=large_dim,
                                                                              # large_dim_head=large_dim//heads,
                                                                              
                                                                              cross_attn_depth=cross_attn_depth, 
                                                                              cross_attn_heads=heads,
                                                                              
                                                                              dropout=dropout))

        self.pool = pool
        # self.to_latent = nn.Identity()
        self.num_classes = num_classes

        self.mlp_head_cxr = nn.Sequential(
            # nn.LayerNorm(small_dim),
            nn.Linear(small_dim, num_classes)
        )

        self.mlp_head_enh = nn.Sequential(
            # nn.LayerNorm(large_dim),
            nn.Linear(large_dim, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, vit_cxr, vit_enh, img_cxr, img_enh):
        
        cxr_ftrs = self.vit_features_cxr(img_cxr) # b, 197, 384
        bs, n, dim = cxr_ftrs.shape
        # xs = xs[:,0] # b, 384
        x_cxr = vit_cxr(img_cxr)
        
        enh_ftrs = self.vit_features_enh(img_enh)
        # print (xl.shape)
        x_enh = vit_enh(img_enh)
        
        for multi_scale_transformer in self.multi_scale_transformers:
            cxr_ca, enh_ca = multi_scale_transformer(cxr_ftrs, enh_ftrs)
            # xs = multi_scale_transformer(xs, xl)
        
        cxr_fus = cxr_ftrs+cxr_ca
        enh_fus = enh_ftrs+enh_ca
        
        cxr_cls = cxr_fus.mean(dim = 1) if self.pool == 'mean' else cxr_fus[:, 0]
        enh_cls = enh_fus.mean(dim = 1) if self.pool == 'mean' else enh_fus[:, 0]

        cxr_ds = self.mlp_head_cxr(cxr_cls)
        enh_ds = self.mlp_head_enh(enh_cls)
        # x = xs + xl
        
        cxr_ds = cxr_ds.view(bs, 1, self.num_classes)
        enh_ds = enh_ds.view(bs, 1, self.num_classes)

        fused_features = torch.cat([cxr_ds, enh_ds], dim=1)
        fused_features = torch.sum(fused_features, dim=1).squeeze(dim=1)
        
        return fused_features, x_cxr, x_enh
    
    
   # cal_out = g_sl(cal_out)

# xl = torch.cat((cal_out, x_large), dim=1)


# # Cross Attn for Smaller Patch
# cal_q = f_sl(small_class.unsqueeze(1))
# cal_qkv = torch.cat((cal_q, x_large), dim=1)

# cal_out = cal_q + cross_attn_s(cal_qkv)

# cal_out_s = g_ls(cal_out)            


# cal_out_l = g_sl(cal_out)
# xl = torch.cat((cal_out, x_large), dim=1) 

# if __name__ == "__main__":
    
#     img = torch.ones([1, 3, 224, 224])
    
#     model = CrossViT(224, 3, 1000)

#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
#     print('Trainable Parameters: %.3fM' % parameters)
    
#     out = model(img)
    
#     print("Shape of out :", out.shape)      # [B, num_classes]

    

# with open('example_PCAM_xq.json') as f:
#     cfg = edict(json.load(f))


# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x



# class Fus_CrossViT(nn.Module):
#     def __init__(self, cfg, model_cnn, model_vit):
#         super().__init__()
        
#         self.cfg = cfg
#         self.vit_features = vit_features(model_vit)
#         small_ftrs = model_vit.head.in_features
#         self.cnn_features = cnn_features(model_cnn)
#         large_ftrs = model_cnn.classifier.in_features
        
#         self.crossvit = CrossViT(cfg, small_dim = small_ftrs, large_dim = large_ftrs)
        

        
#     def forward(self, cxr, enh):
        
#         xs = self.vit_features(enh)
#         # print('xs_input:',  len(xs), xs[-1].shape)
#         xl = self.cnn_features(cxr)
#         # print (xl.shape)
#         x, xs, xl= self.crossvit(xs, xl)
        
#         return x, xs, xl


# class CrossViT(nn.Module):
#     def __init__(self, cfg, small_dim = 768,
#                  large_dim = 1024, cross_attn_depth = 1, multi_scale_enc_depth = 3,
#                  heads = 3, pool = 'cls', dropout = 0., dist = None):
#         super().__init__()


#         self.multi_scale_transformers = nn.ModuleList([])
        
#         for _ in range(multi_scale_enc_depth):
#             self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim,
#                                                                               small_dim_head=small_dim//heads,
#                                                                               # small_mlp_dim=small_dim*scale_dim,
                                                                              
#                                                                               large_dim=large_dim,
                                                                              
#                                                                               cross_attn_depth=cross_attn_depth, 
#                                                                               cross_attn_heads=heads, 
#                                                                               dropout=dropout))

#         self.dist = dist                                                                      
#         self.cfg = cfg
#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head_small = ClassificationHead(self.cfg, small_dim)
#         self.mlp_head_large = ClassificationHead(self.cfg, large_dim)


#     def forward(self, xs, xl):
#         # small = cxr
#         if self.dist:
#             cls_token, dist_token = xs[0], xs[1]
#             xs_new = ((cls_token+dist_token)/2, xs[-1])
#         else:
#             xs_new = (xs[0], xs[-1])
        
#         # print (len(xs_new), xs_new[0].shape, xs_new[-1].shape)
#         # large = enh
#         # b, dim, H, W = xl.shape
#         # xl_new = rearrange(xl, 'b e h w -> b (h w) e', h=H, w=W)
#         # print (xl_new.shape)
#         xl_new = xl
        
#         for multi_scale_transformer in self.multi_scale_transformers:
#             xs, xl = multi_scale_transformer(xs_new, xl_new)
        
#         xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
#         xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]

#         xs = self.mlp_head_small(xs)
#         xl = self.mlp_head_large(xl)
#         x = xs + xl
        
#         return x, xs, xl


# class ClassificationHead_Small(nn.Module):
#     def __init__(self, cfg, small_dim):
#         super(ClassificationHead_Small, self).__init__()
#         self.cfg = cfg
#         self.norm = nn.LayerNorm(small_dim, eps=1e-6)
#         self.embed_dim = small_dim
        
#         for index, num_class in enumerate(self.cfg.num_classes):
#             setattr(self, "fc_" + str(index), nn.Linear(small_dim,num_class))

#     def forward(self, x):
#         feat_map = self.norm(x)
#         logits = list()
#         for index, num_class in enumerate(self.cfg.num_classes):
#             classifier = getattr(self, "fc_" + str(index))
#             logit = classifier(feat_map)
#             logits.append(logit)

#         return logits