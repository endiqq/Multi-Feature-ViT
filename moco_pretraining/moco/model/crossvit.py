import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward, CrossAttention
import numpy as np
import einops

# from bin.cxr_arm_models import CXR_Arm_ViT_w_classifier as CXR_Arm_ViT
# from bin.enh_arm_models import Enh_Arm_ViT_w_classifier as Enh_Arm_ViT
# from model.crossvit import crossvit


# class ClassificationHead(nn.Module):
#     def __init__(self, cfg, large_dim):
#         super(ClassificationHead, self).__init__()
#         self.cfg = cfg
#         self.norm = nn.LayerNorm(large_dim, eps=1e-6)
#         self.embed_dim = large_dim
        
#         for index, num_class in enumerate(self.cfg.num_classes):
#             setattr(self, "fc_" + str(index), nn.Linear(large_dim,num_class))

#     def forward(self, x):
#         feat_map = self.norm(x)
#         logits = list()
#         for index, num_class in enumerate(self.cfg.num_classes):
#             classifier = getattr(self, "fc_" + str(index))
#             logit = classifier(feat_map)
#             logits.append(logit)

#         return logits
    
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(self, small_dim = 384, #small_depth = 4, small_heads =3, small_dim_head = 32, small_mlp_dim = 384,
                 large_dim = 512,  large_dim_head = 64, #large_depth = 1, large_heads = 3,large_mlp_dim = 768,
                 cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                
                # nn.Linear(large_dim, small_dim),
                # nn.Linear(small_dim, large_dim),
                # PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
                
            ]))

    def forward(self, xs, xl):
        # print (len(xs), xs[0].shape)
        # xs = self.transformer_enc_small(xs)
        # xl = self.transformer_enc_large(xl)

        # for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
        for f_sl, g_ls, cross_attn_s in self.cross_attn_layers:          
            small_class = xs[:,0]
            x_small = xs[:, 1:]
            # print (x_small.shape)
            x_large = xl
            # print (x_large.shape)
            # large_class  =xl
                        


            # # Cross Attn for Large Patch

            # cal_q = f_ls(large_class.unsqueeze(1))
            # cal_qkv = torch.cat((cal_q, x_small), dim=1)
            
            # cal_out_s = cal_q + cross_attn_l(cal_qkv)
            
            # xs = torch.cat((cal_out_s, x_small), dim=1)
            # xs_mix = torch.cat(((cal_out_s+small_class)/2, x_small), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            # xs = torch.cat((cal_out, x_small), dim=1)

        return cal_out#xs, xl


# class vit_features(nn.Module):
#     def __init__(self, model, distilled = None):
#         super(vit_features, self).__init__()
        
#         self.dist = distilled
#         self.patch_embed = model.patch_embed
#         self.cls_token = model.cls_token
#         if self.dist is not None:
#             self.dist_token = model.dist_token
#         self.pos_drop = model.pos_drop
#         # self.blocks = model.blocks[0:10]
#         self.blocks = model.blocks
#         self.norm = model.norm
        
#         self.pos_embed = model.pos_embed
        
#     def forward_features(self, x):

#         for blk in self.blocks:
#             x = blk(x)          
#         return x
    
#     def forward(self, x):
#         x = self.patch_embed(x)
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
#         if self.dist is None:
#             x = torch.cat((cls_token, x), dim=1)
#         else:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.norm(self.forward_features(x))
#         # x = rearrange(x, 'b (h w) c -> b c h w', h=16, w=16)
#         if self.dist is None:
#             # return self.pre_logits(x[:, 0])
#             return x[:, 0], x[:, 1:]
#         else:
#             return x[:, 0], x[:, 1], x[:, 2:]
        
# # print (len(vit_features(vit)(test_img)), vit_features(vit)(test_img)[0].shape, vit_features(vit)(test_img)[1].shape)

# class cnn_features(nn.Module):
#     def __init__(self, model):
#         super(cnn_features, self).__init__()
        
#         self.model = model
#     def forward(self, x):
#         features = self.model.features(x)
        
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
        
#         return out
        
   
# vit_b = 768; vit_s = 384    
class Fus_CrossViT(nn.Module):
    def __init__(self, model_cnn, model_vit, small_dim = 384,
                 large_dim = 512, cross_attn_depth = 1, multi_scale_enc_depth = 1,
                 heads = 3, dropout = 0.):
        super().__init__()
        
        # self.cfg = cfg
        self.vit_features = model_vit.features3D
        # small_ftrs = model_vit.head.in_features
        
        self.cnn_features = nn.Sequential(*list(model_cnn.children()))[:-2]
        # large_ftrs = model_cnn.classifier.in_features
        
        self.multi_scale_transformers = nn.ModuleList([])
        
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(MultiScaleTransformerEncoder(small_dim=small_dim,
                                                                              # small_dim_head=small_dim//heads,
                                                                              # small_mlp_dim=small_dim*scale_dim,
                                                                              
                                                                              large_dim=large_dim,
                                                                              large_dim_head=large_dim//heads,
                                                                              cross_attn_depth=cross_attn_depth, 
                                                                              cross_attn_heads=heads, 
                                                                              dropout=dropout))

        # self.dist = dist                                                                      
        # self.cfg = cfg
        # self.pool = pool
        # self.to_latent = nn.Identity()
        
        # self.pos_embedding_small = nn.Parameter(torch.randn(1, 196 + 1, small_dim))
        # self.dropout_small = nn.Dropout(dropout)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(small_dim),
            nn.Linear(small_dim, 3)
        )

        
    def forward(self, img):
        
        xs = self.vit_features(img) # b, 197, 384
        b,n,dim = xs.shape
        # xs = xs[:,0] # b, 384
        
        xl = self.cnn_features(img)
        # print (xl.shape)
        # l_dim = xl.shape[1]
        xl = einops.rearrange(xl, 'b c h w -> b (h w) (c)') #b 49, 512
        
        # if self.dist:
        #     cls_token, dist_token = xs_enh[0], xs_enh[1]
        #     cls_token_small = ((cls_token+dist_token)/2).unsqueeze(1)
                        
        # else:
            
        #     cls_token_small = xs_enh[0].unsqueeze(1)
            
        # xs = torch.cat((cls_token_small, xs_enh[-1]), dim=1)
        # xs += self.pos_embedding_small[:, :(n + 1)]
        # xs = self.dropout_small(xs)
        
        # xl = xl_cxr
        
        for multi_scale_transformer in self.multi_scale_transformers:
            # xs, xl = multi_scale_transformer(xs, xl)
            xs = multi_scale_transformer(xs, xl)
        
        # xs = xs.mean(dim = 1) if self.pool == 'mean' else xs[:, 0]
        # xl = xl.mean(dim = 1) if self.pool == 'mean' else xl[:, 0]
        # print (xs.shape)
        logits = self.mlp_head(xs.squeeze(1))
        # xl = self.mlp_head_large(xl)
        
        # x = [xs[i] + xl[i] for i in range(len(self.cfg.num_classes))]
        
        return logits #x, xs, xl
    
    
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