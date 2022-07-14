import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,q,k,v


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_norm=self.norm1(x)
        x_att,q,k,v=self.attn(x_norm)
        x = x + self.drop_path(x_att)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,q,k,v


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t,q,k,v = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t,q,k,v


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        conv_features=[]
        tran_features=[]
        q=[]
        k=[]
        v=[]

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        #print('x_base',x_base.shape)
        conv_features.append(x_base)

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)
        conv_features.append(x)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        #print('x_t flatten',x_t.shape)
        tran_features.append(x_t)
       
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        #print('x_t n tokens',x_t.shape)
        x_t,q1,k1,v1 = self.trans_1(x_t)
        #print('x_t tran_1 q k  v',x_t.shape,q1.shape,k1.shape,v1.shape)
        tran_features.append(x_t)
        q.append(q1)
        k.append(k1)
        v.append(v1)
        
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_t,qi,ki,vi = eval('self.conv_trans_' + str(i))(x, x_t)
            conv_features.append(x)
            tran_features.append(x_t)
            q.append(qi)
            k.append(ki)
            v.append(vi)

        
        return conv_features,tran_features,q,k,v

class JLModule(nn.Module):
    def __init__(self, backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone
        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        

    def forward(self, x):

        conv,tran,q,k,v = self.backbone(x)
        '''for i in range(len(conv)):
            print(i,"     ",conv[i].shape,tran[i].shape)'''
        

        return conv,tran,q,k,v # list of tensor that compress model output

class LDELayer(nn.Module):
    def __init__(self):
        super(LDELayer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_c=nn.Sequential(nn.Conv2d(256, 64, 1, 1, 1), self.relu, nn.Conv2d(64, 64, 3, 1, 0), self.relu)
        #self.conv_d=nn.Sequential(nn.MaxPool2d(3),nn.Conv2d(256, 64, 7, 1, 6), nn.Conv2d(64, 64, 7, 1, 2), self.relu)
        self.conv_d1=nn.MaxPool2d(3,1,1)
        self.conv_d2=nn.Conv2d(256, 64, 7, 1, 3)
        self.conv_d3= nn.Conv2d(64, 64, 7, 1, 3)


    def forward(self, list_x,list_y):
        #fconv_c=[]
        #fconv_d=[]
        result=[]
        tran_c=[]
        
        '''for i in range(len(list_x)):
            rgb_conv = list_x[i][0]
            depth_conv = list_x[i][1]
            rgb_tran = list_y[i][0]
            depth_tran = list_y[i][1]
            print("******LDE layer******")
            print(i,"     ",rgb_conv.shape,rgb_tran.shape,depth_tran.shape)'''
        for j in range(1,5):
            fconv_c=self.conv_c((list_x[j][0]).unsqueeze(0))
            a=self.conv_d1((list_x[j][1]).unsqueeze(0))  
            b=self.conv_d2(a)
            fconv_d=self.relu(self.conv_d3(b))
            sum_t_lde=(list_y[j][0]+list_y[j][1]).unsqueeze(0)
            mul_t_lde=(list_y[j][0]*list_y[j][1]).unsqueeze(0)
            tran_c.append(torch.cat((sum_t_lde,mul_t_lde),dim=0))
            sum=torch.cat((fconv_c, fconv_d), dim=0)
            result.append(sum)
            #print('LDE conv tran output',sum.shape,tran_c[0].shape)
            
            

        return result,tran_c

class CoarseLayer(nn.Module):
    def __init__(self):
        super(CoarseLayer, self).__init__()
        self.score = nn.Conv2d(1024, 1, 1, 1)
        self.co=nn.Conv1d(401,64,3,3,1)
        

    def forward(self, x, y):
        
        _,_,H,W=x.shape
        x_sal = self.score(x)
        y_sal= (self.co(y)).view(1,1,128,128)
        x_sal = F.interpolate(x_sal, 320, mode='bilinear', align_corners=True)
        y_sal = F.interpolate(y_sal, 320, mode='bilinear', align_corners=True)
        
        return x_sal+y_sal+(x_sal*y_sal)

class GDELayer(nn.Module):
    def __init__(self):
        super(GDELayer, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1024=nn.Conv2d(1024,1,1,1)
        self.conv512=nn.Conv2d(512,1,1,1)
        

    def forward(self, x, y,coarse_sal):
        w,h=coarse_sal.size(2),coarse_sal.size(3)
        out_RA=[]
        gde_t=[]
        for j in range(12,4,-1):
            rgb_part=(x[j][0]).unsqueeze(0)
            depth_part=(x[j][1]).unsqueeze(0)
            #print('shape of y',y[j][0].shape,y[j][1].shape)
            if (rgb_part.size(2)!= coarse_sal.size(2)) or (rgb_part.size(3) != coarse_sal.size(3)):
                rgb_part = F.interpolate(rgb_part, w, mode='bilinear', align_corners=True)
                depth_part = F.interpolate(depth_part, w, mode='bilinear', align_corners=True)
            salr=self.sigmoid(coarse_sal[0])
            Ar=1-salr
            rgb_att=Ar*rgb_part
            sald=self.sigmoid(coarse_sal[1])
            Ad=1-sald
            depth_att=Ad*depth_part
            c_att = torch.cat((rgb_att, depth_att), dim=0)
            if (rgb_part.size(1)==1024):
                out_RA.append(self.conv1024(c_att))
            else:
                out_RA.append(self.conv512(c_att))
            #print('conv gde out',out_RA[0].shape)
            sum_t_gde=(y[j][0]+y[j][1]).unsqueeze(0)
            mul_t_gde=(y[j][0]*y[j][1]).unsqueeze(0)
            gde_t.append(torch.cat((sum_t_gde,mul_t_gde),dim=0))
            #print('tran gde out',gde_t[0].shape)

                
            

        return out_RA,gde_t

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample=nn.ConvTranspose2d(64, 1, kernel_size=3, stride=4, padding=1, output_padding=3,dilation=1)
        self.softmax=nn.Softmax()
        


    def forward(self, lde_c,gde_c,lde_t,gde_t,q,k,v):
        low_features_conv=[]
        high_features_conv=[]
        low_features_tran=[]
        high_features_tran=[]
        lfc=torch.zeros(2,1,320,320)
        gfc=torch.zeros(2,1,320,320)
        lft=torch.zeros(2,1,320,320)
        gft=torch.zeros(2,1,320,320)
        for a in range(len(q)):
            q[a]=q[a].permute(0,2,1,3).flatten(2)
            k[a]=k[a].permute(0,2,1,3).flatten(2)
            v[a]=v[a].permute(0,2,1,3).flatten(2)
            #print('shape of q',q[a].shape)
        l_index=0
        h_index=4
        for j in range(len(lde_c)):
            #print('decoder lde_c',lde_c[j].shape)
            lde_c[j]=self.upsample(lde_c[j])
            #print('decoder lde_c after upsample',lde_c[j].shape,lde_c[j][0].shape,lde_c[j][1].shape)
            sum_low=(lde_c[j][0] + lde_c[j][1]).unsqueeze(0)
            mul_low=(lde_c[j][0] * lde_c[j][1]).unsqueeze(0)
            #print('sumlow mullow',sum_low.shape,mul_low.shape)
            lfc=torch.cat((sum_low,mul_low), dim=0)
            #print('lfc',lfc.shape)
            low_features_conv.append(lfc)
            lfc+=lfc
            tran_low1=(lde_t[j][1]*(self.softmax(q[l_index][1]*k[l_index][0])*v[l_index][0])).unsqueeze(0)
            tran_low2=(lde_t[j][0]*(self.softmax(q[l_index][0]*k[l_index][1])*v[l_index][1])).unsqueeze(0)
            #print('tran low1 2',tran_low1.shape,tran_low2.shape)
            cat_tran_low=torch.cat((tran_low1,tran_low2),dim=0)
            cat_tran_low=cat_tran_low.unsqueeze(1)
            lft=F.interpolate(cat_tran_low, (320,320), mode='bilinear', align_corners=True)
            low_features_tran.append(lft)
            lft+=lft
            
            l_index=l_index+1
        for k1 in range(len(gde_c)):
            sum_high=(gde_c[k1][0] + gde_c[k1][1]).unsqueeze(0)
            mul_high=(gde_c[k1][0] * gde_c[k1][1]).unsqueeze(0)
            gfc=torch.cat((sum_high,mul_high), dim=0)
            high_features_conv.append(gfc)
            gfc+=gfc
            
            tran_high1=(gde_t[k1][1]*(self.softmax(q[h_index][1]*k[h_index][0])*v[h_index][0])).unsqueeze(0)
            tran_high2=(gde_t[k1][0]*(self.softmax(q[h_index][0]*k[h_index][1])*v[h_index][1])).unsqueeze(0)
            cat_tran_high=torch.cat((tran_high1,tran_high2),dim=0)
            cat_tran_high=cat_tran_high.unsqueeze(1)
            gft=F.interpolate(cat_tran_high, (320,320), mode='bilinear', align_corners=True)
            high_features_tran.append(gft)
            gft+=gft
            h_index=h_index+1
            #print('ok too')
        
        '''for m in range(7):
            print('high_features_conv',high_features_conv[m].shape)
            print('high_features_tran',high_features_tran[m].shape)
        for m in range(3):
            print('low_features_conv',low_features_conv[m].shape)
            
            print('low_features_tran',low_features_tran[m].shape)'''
            
        #print(lfc.shape,lft.shape,gft.shape,gfc.shape,len(lfc))
        return lfc,lft,gft,gfc

class JL_DCF(nn.Module):
    def __init__(self,JLModule,lde_layers,coarse_layer,gde_layers,decoder):
        super(JL_DCF, self).__init__()
        
        self.JLModule = JLModule
        self.lde = lde_layers
        self.coarse_layer=coarse_layer
        self.gde_layers=gde_layers
        self.decoder=decoder
        self.final_conv=nn.Conv2d(8,1,1,1,0)
        
    def forward(self, f_all):
        x,y,q,k,v = self.JLModule(f_all)
        lde_c,lde_t = self.lde(x,y)
        coarse_sal=self.coarse_layer(x[12],y[12])
        gde_c,gde_t=self.gde_layers(x,y,coarse_sal)
        '''print('lde_c',lde_c[0].shape,len(lde_c))
        print('lde_t',lde_t[0].shape,len(lde_t))
        print('gde_c',gde_c[0].shape,len(gde_c))
        print('gde_t',gde_t[0].shape,len(gde_t))
        print('coarse_sal',coarse_sal.shape)

        for i in range(len(q)):
            print('q',q[i].shape,len(q))
            print('k',k[i].shape,len(k))
            print('v',v[i].shape,len(v))'''
        sal_lde_conv,sal_lde_tran,sal_gde_conv,sal_gde_tran=self.decoder(lde_c,gde_c,lde_t,gde_t,q,k,v)
        final=torch.cat((sal_lde_conv[0],sal_lde_tran[0],sal_gde_conv[0],sal_gde_tran[0],sal_lde_conv[1],sal_lde_tran[1],sal_gde_conv[1],sal_gde_tran[1]),dim=0)
        #print('finalcat',final.shape)
        sal_final=self.final_conv((final).unsqueeze(0))
        #print('sal_final',sal_final.shape)
        
        return sal_final,sal_lde_conv,sal_lde_tran,sal_gde_conv,sal_gde_tran,coarse_sal

def build_model(network='conformer', base_model_cfg='conformer'):
   
        backbone= Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True)
        
   

        return JL_DCF(JLModule(backbone),LDELayer(),CoarseLayer(),GDELayer(),Decoder())
