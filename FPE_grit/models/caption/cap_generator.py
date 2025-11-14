import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from models.common.attention import MultiHeadAttention, Attention, MultiHeadAttention_w_weights
from models.common.pos_embed import sinusoid_encoding_table, FeedForward
from models.caption.containers import Module, ModuleList
from models.caption.FPE_module import Freq_Perturbation_Entropy


class GeneratorLayer(Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__()

        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_ff, dropout)


class ParallelAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, activation='sigmoid', n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        
        self.vis_att1 = MultiHeadAttention_w_weights(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories) # grid
        self.vis_att2 = MultiHeadAttention_w_weights(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):

        # (b_s, seq_len, d_model) <--- (b_s, seq_len, d_model)   (b_s, 1, seq_len, seq_len)
        self_att = self.self_att(x, x, x, mask_x) 
        # (b_s, seq_len, d_model) <--- (b_s, seq_len, d_model)   (b_s, seq_len, 1)
        self_att = self_att * mask_pad  

        #                   [b, seq_len, d_model] [b, (h w), d_model]  [b, 1, 1, (h w)]  
        output1 = self.vis_att1(self_att, y1, y1, mask_y1)
        # [b, seq_len, d_model] [b, seq_len, (h w)]
        enc_att1, weights_gri  = output1[0] * mask_pad, output1[1]
        
        output2 = self.vis_att2(self_att, y2, y2, mask_y2)
        enc_att2, weights_reg   = output2[0] * mask_pad, output2[1]


        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))  #  fc_alpha1
    
        
        # (b_s, seq_len, d_model) <--- (b_s, seq_len, d_model)x(b_s, seq_len, d_model)  
        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2) / np.sqrt(2)   
        enc_att = enc_att * mask_pad  #  (b_s, seq_len, d_model) <--- (b_s, seq_len, d_model)*[b_s, seq_len, 1]
        ff = self.pwff(enc_att)  # (b_s, seq_len, d_model)  FeedForward
        ff = ff * mask_pad  #  (b_s, seq_len, d_model) <--- (b_s, seq_len, d_model)*[b_s, seq_len, 1]
        return ff, [weights_gri, weights_reg]


class ConcatAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

    def forward(self, x, y, mask_pad, mask_x, mask_y):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att(out, y, y, mask_y) * mask_pad
        out = self.pwff(out) * mask_pad
        return out


class SequentialAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention_w_weights(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention_w_weights(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        out = self.self_att(x, x, x, mask_x) * mask_pad


        output1 = self.vis_att1(out, y1, y1, mask_y1) 
        enc_att1, weights_gri  = output1[0] * mask_pad, output1[1]
        output2 = self.vis_att1(enc_att1, y2, y2, mask_y2) 
        enc_att2, weights_reg  = output2[0] * mask_pad, output2[1]
        

        ff = self.pwff(enc_att2)
        ff = ff * mask_pad
        return ff, [weights_gri, weights_reg]


class CaptionGenerator(Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'parallel': ParallelAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }

    def __init__(self,
                 vocab_size,
                 max_len, # 54
                 n_layers,
                 pad_idx,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 dropout=.1,
                 decoder_name='parallel',
                 cfg=None):
        super().__init__()
        
        self.d_model = d_model
        # 词嵌入
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.cfg_all=cfg
        self.cfg = cfg.model.cap_generator
        self.decoder_name = self.cfg.decoder_name
        # self.decoder_name = 'sequential'
        # print(f"================in {self.decoder_name}==================")
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]

        self.layers = ModuleList([generator_layer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers

        self.gamma=self.cfg_all.model.freq_net.gamma
        self.visualization=self.cfg_all.model.freq_net.visualization


        self.FPE = Freq_Perturbation_Entropy(
            visual_type=self.cfg_all.model.freq_net.visual_type,
            cfg=self.cfg_all.model.freq_net,
            )


        self.register_state('running_mask_x', torch.zeros((1, 1, 0)).byte()) # running_mask_x  
        self.register_state('running_seq', torch.zeros((1,)).long())   # ，running_seq  
        self.register_state('running_mask_x_ca', torch.zeros((1, 1, 0)).byte()) # mask for cross atten

    def get_seq_inputs(self, input):
        # input.shape (b_s, seq_len);   when use beam search in SC : input.shape  [Bs*Beam, 1] 
        b_s, seq_len = input.shape[:2]
        

        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)  # (seq_len, seq_len)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()  # (b_s, 1, 1, seq_len)  
        mask_x = mask_x.gt(0)   
        mask_x_ca= (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()  #mask for crossattention (b_s, 1, 1, seq_len)
        mask_x_ca = mask_x_ca.gt(0)


        if self._is_stateful:  # XE：False      SC：True
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x
           
            self.running_mask_x_ca =  torch.cat([self.running_mask_x_ca, mask_x_ca],-1)
            mask_x_ca= self.running_mask_x_ca

        mask_pad = (input != self.pad_idx).unsqueeze(-1).float() 

        
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)

        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)  # (b_s, seq_len)

        if self._is_stateful: 
            self.running_seq.add_(1)
            seq = self.running_seq
        
        #   (b_s, seq_len, d_model) +  (b_s, seq_len,, d_model)
        x = self.word_emb(input) + self.pos_emb(seq)

        return x, mask_x, mask_pad, mask_x_ca



    def forward(self, input, vis_inputs, images_freq_t, images_freq_m):

        x, mask_x, mask_pad, mask_x_ca = self.get_seq_inputs(input)  

        if self.cfg_all.model.use_freq_feat: 
       
            mu, variance, feat_mask  = self.FPE( x , images_freq_t, images_freq_m, mask_pad, mask_x, mask_x_ca )
            #  torch.Size([bs, num, h, w, d_mode])  <------  [B, h, w, d_mode]    
            freq_perturbation = self.FPE.sample(mu, variance) 

            if self.visualization:
                freq_perturbation = freq_perturbation.repeat(1, 1, 1, self.d_model)   #  [bs, h, w, 512 ]  <-  [bs, h, w, 1] 
                vis_inputs['gri_feat'] = vis_inputs['gri_feat'] + self.gamma * rearrange(freq_perturbation, 'b h w c -> b (h w) c')         
            else:
                freq_perturbation = freq_perturbation    
                vis_inputs['gri_feat'] = vis_inputs['gri_feat']  + self.gamma * rearrange(freq_perturbation, 'b h w c -> b (h w) c')         
                
            perturb = freq_perturbation 


        if self.decoder_name == 'concat':
            y = torch.cat([vis_inputs['grid_feat'], vis_inputs['reg_feat']], dim=1)
            mask_y = torch.cat([vis_inputs['gri_mask'], vis_inputs['reg_mask']], dim=3)

            for layer in self.layers:
                x = layer(x, y, mask_pad, mask_x, mask_y)

        if self.decoder_name == 'sequential':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x, attention_weights_list = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        
        if self.decoder_name == 'parallel':

            y1 = vis_inputs['gri_feat']     
            y2 = vis_inputs['reg_feat']     

            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x, attention_weights_list = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        

        reference_point = vis_inputs['reg_point'] # [Batch, num_queries, 4]
        attention_weights_list.append(reference_point)

        attention_weights_list.append(perturb)

        x = self.fc(x)

        return F.log_softmax(x, dim=-1), attention_weights_list
