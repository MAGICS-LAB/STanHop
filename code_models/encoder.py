import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from code_models.hopfield import Hopfield, STHMLayer
from math import ceil

class CoarseGrain(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.coarse_grain = CoarseGrain(d_model, win_size, nn.LayerNorm)
        else:
            self.coarse_grain = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(STHMLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout))
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.coarse_grain is not None:
            x = self.coarse_grain(x)
        
        for layer in self.encode_layers:
            x = layer(x)        
        
        return x

class Encoder(nn.Module):
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x