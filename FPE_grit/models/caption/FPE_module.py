import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from models.common.attention import MultiHeadAttention, Attention_fpe, MultiHeadAttention_wo_residual
from models.common.pos_embed import sinusoid_encoding_table, FeedForward
from models.caption.containers import Module, ModuleList
import math
from torch.nn.functional import normalize
from math import sqrt 
from math import ceil

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50_ImageEncoder(nn.Module):
    """
    ResNet50 
    """

    def __init__(self, d_model=512):
        super(ResNet50_ImageEncoder, self).__init__()
        self.d_model = d_model
        self.in_channels = 64

        # 384 -> 192 (conv stride=2) -> 96 (maxpool stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)



        self.channel_proj = nn.Sequential(
            nn.Conv2d(2048, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):

        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []

        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        x = self.conv1(x)   
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     

        x = self.layer1(x)     
        x = self.layer2(x)     
        x = self.layer3(x)    
        x = self.layer4(x)     
        x = self.channel_proj(x)  

        return x



class ViT_ImageEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, num_layers=4):
        super(ViT_ImageEncoder, self).__init__()
        self.d_model = d_model
        self.patch_size = 32
        self.max_seq_len = 12 * 20   
 
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=self.patch_size, 
                                     stride=self.patch_size, padding=0)
        
 
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
 
 
        assert x.size(2) == 384, f"hight must be 384, but is {x.size(2)}"
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch, d_model, h, w]
        _, _, h, w = x.size()
        assert h == 12, f" path in cloum must be 12, but is{h}"
        
 
        L = h * w
        x = x.flatten(2).permute(0, 2, 1)  # [batch, L, d_model]
        
 
        if L < self.max_seq_len:
            padding = torch.zeros(x.size(0), self.max_seq_len - L, x.size(2), device=x.device)
            x = torch.cat([x, padding], dim=1)
        x = x + self.pos_embed
        x = self.norm(x)
        
 
        key_padding_mask = None
        if mask is not None:
 
            mask = mask.float()
            
 
            patch_h = x.size(2) // self.patch_size
            patch_w = (x.size(3) + self.patch_size - 1) // self.patch_size
            
 
            mask = F.adaptive_avg_pool2d(mask, (patch_h, patch_w))
 
            mask_seq = mask.view(mask.size(0), -1)  # [batch, patch_h * patch_w]
            
  
            key_padding_mask = torch.zeros(x.size(0), self.max_seq_len, dtype=torch.bool, device=x.device)
            
            actual_L = min(patch_h * patch_w, self.max_seq_len)
            

            if mask_seq.size(1) >= actual_L:
                key_padding_mask[:, :actual_L] = mask_seq[:, :actual_L] == 0
            else:
                key_padding_mask[:, :mask_seq.size(1)] = mask_seq == 0
                
            if actual_L < self.max_seq_len:
                key_padding_mask[:, actual_L:] = True
        
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x[:, :L, :] if L < self.max_seq_len else x
        x = x.permute(0, 2, 1).view(-1, self.d_model, h, w)
        return x
        

class Freq_Perturbation_Entropy(Module):

    def __init__(self,
                visual_type,
                cfg=None,
                d_model=512, 
                n_heads=8, 
                d_ff=2048, 
                dropout=.1):
        super(Freq_Perturbation_Entropy, self).__init__()

        self.visual_type = visual_type
        self.visualization = cfg.visualization

        if self.visualization:
            self.out_channel = 1 #  1 
        else:
            self.out_channel = d_model #  d_model


        self.text_encoder = MultiHeadAttention_wo_residual(d_model, n_heads, dropout, can_be_stateful=True)  # for sqe encoder in fpe
        self.cross_attenton = Attention_fpe(d_model, n_heads, dropout)   
        
        self.fc_variance = nn.Linear(d_model, d_model)
        self.fc_mean = nn.Linear(d_model, d_model)

        self.conv_variance = nn.Conv2d(d_model, self.out_channel, kernel_size=1,stride=1, padding=0, bias=False)
        self.conv_mean = nn.Conv2d(d_model, self.out_channel, kernel_size=1,stride=1, padding=0, bias=False)

        
        if self.visual_type == "simple_CNN":

            self.img_encoder = nn.Sequential(

                nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),  
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2), 
                nn.MaxPool2d(kernel_size=2, stride=2),  
                nn.BatchNorm2d(64),
                

                nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                

                nn.Conv2d(256, d_model, kernel_size=1), 
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
        elif self.visual_type == "ViT":
            self.img_encoder = ViT_ImageEncoder(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                num_layers=cfg.num_layers
            )
        elif self.visual_type == "CNN":
            self.img_encoder = ResNet50_ImageEncoder(d_model=d_model)

        else:
            raise ValueError(f"Unsupported visual_type: {self.visual_type}")


    def forward(self, seq, freq_imag_tensor, freq_imag_mask, mask_pad_seq, mask_seq, mask_x_ca):

        #########################
        # FPE
        #########################
        
        text_feat = self.text_encoder(seq, seq, seq, mask_seq, mask_pad_seq)
        freq_tensors, freq_mask = freq_imag_tensor, freq_imag_mask  # nest tensor


        original_width = freq_tensors.size(-1)
        target_width = math.ceil(original_width / 32)  
        padded_pixel_width = target_width * 32  
        pad_right = max(0, padded_pixel_width - original_width)
        if pad_right > 0:
            freq_tensors = F.pad(freq_tensors, (0, pad_right, 0, 0), mode='constant', value=0)

        if self.visual_type == "CNN":
            freq_feat_initial = self.img_encoder(freq_tensors)   
        elif self.visual_type == "ViT":
            freq_feat_initial = self.img_encoder(freq_tensors, freq_mask)   

        freq_mask_initial = F.interpolate(freq_mask[:, None, :, :].float(), size=freq_feat_initial.shape[-2:]).squeeze(1).bool()   # masks [B, h, w]   
        img_feat = rearrange(freq_feat_initial, 'b c h w -> b (h w) c')  
        freq_mask = rearrange(freq_mask_initial, 'b h w -> b (h w) 1')      
        
 
        # [B, h*w, d_model]          [B, h*w, d_model]     (b_s, seq_len, d_model)  or  (b_s, step,  d_model)     (b_s, 1, seq_len, seq_len)    
        text_freq_feat = self.cross_attenton(img_feat, text_feat, text_feat, mask_x_ca)  
        
        # [B, h*w, d_model]     [B, h*w, d_model] # [B,  h*w，1]
        text_freq_feat= text_freq_feat * freq_mask
        # text_freq_feat= text_freq_feat
        
        _, _, hi, wi = freq_feat_initial.shape
        text_freq_feat = rearrange(text_freq_feat, 'b (h w) d_model -> b d_model h w ', h=hi, w=wi) # [B, d_model, h, w]
        text_freq_feat_mask =  rearrange(freq_mask_initial, 'b h w -> b 1 h w') # [B, 1, h, w]

        variance = (self.conv_variance(text_freq_feat)).permute(0, 2, 3, 1)  
        mu = (self.conv_mean(text_freq_feat)).permute(0, 2, 3, 1)  
                            
        return mu, variance,  []

    def sample(self, mu, variance):
        
        var = variance  
        m = mu
        perturbation = torch.randn_like(var) 
        perturbation = var * perturbation + m

        return perturbation
        


    
