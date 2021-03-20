import torch
import matplotlib.pyplot as plt
import torchvision
from initialize import *

# https://github.com/ozan-oktay/Attention-Gated-Networks
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features) - 1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i + 1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i + 1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op             = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l + g)  # batch_sizex1xWxH

        if self.normalize_attn:
            a = torch.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)

        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            g = torch.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

        return c.view(N, 1, W, H), g

'''class GridAttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=False):
        super(GridAttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = torch.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)

        c = self.phi(torch.relu(l_ + g_))  # batch_sizex1xWxH

        # compute attn map
        if self.normalize_attn:
            a = torch.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)

        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)  # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)  # weighted sum
        else:
            output = torch.adaptive_avg_pool2d(f, (1, 1)).view(N, C)

        return c.view(N, 1, W, H), output'''

#attention max-pooling
class AttnVGG(nn.Module):
    def __init__(self,
                 img_size,
                 num_classes,
                 isAttention=True,
                 normalize_attn=True,
                 attn_before=True,
                 init='xavierUniform'):
        super(AttnVGG, self).__init__()
        self.isAttention    = isAttention
        self.attn_before    = attn_before
        # conv blocks
        self.conv_block1    = ConvBlock(3, 64, 2)
        self.conv_block2    = ConvBlock(64, 128, 2)
        self.conv_block3    = ConvBlock(128, 256, 3)
        self.conv_block4    = ConvBlock(256, 512, 3)
        self.conv_block5    = ConvBlock(512, 512, 3)
        self.conv_block6    = ConvBlock(512, 512, 2, pool=True)
        self.dense          = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(img_size/32), padding=0, bias=True)

        # Projectors & Compatibility functions
        if self.isAttention:
            self.projector  = ProjectorBlock(256, 512)
            self.attn1      = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2      = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3      = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)

        # final classification layer
        if self.isAttention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")

    '''def NormalizeImg(self, img):
        nimg = (img - img.min()) / (img.max() - img.min())
        return nimg
    def show_MNIST(self, img):
        grid    = torchvision.utils.make_grid(img)
        trimg   = grid.numpy().transpose(1, 2, 0)
        plt.imshow(trimg)
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()'''

    def forward(self, data):
        #nimg = self.NormalizeImg(data).detach().cpu()
        #self.show_MNIST(data)
        output  = self.conv_block1(data)
        output  = self.conv_block2(output)

        l1_1    = self.conv_block3(output) # /1
        l1_2    = torch.max_pool2d(l1_1, kernel_size=2, stride=2, padding=0) # /2

        l2_1    = self.conv_block4(l1_2) # /2
        l2_2    = torch.max_pool2d(l2_1, kernel_size=2, stride=2, padding=0) # /4

        l3_1    = self.conv_block5(l2_2) # /4
        l3_2    = torch.max_pool2d(l3_1, kernel_size=2, stride=2, padding=0) # /8

        output  = self.conv_block6(l3_2) # /32
        g       = self.dense(output) # batch_sizex512x1x1

        # pay isAttention
        if self.isAttention:
            if self.attn_before:
                p       = self.projector(l1_1)
                c1, g1  = self.attn1(p, g)
                c2, g2  = self.attn2(l2_1, g)
                c3, g3  = self.attn3(l3_1, g)
            else:
                p       = self.projector(l1_2)
                c1, g1  = self.attn1(p, g)
                c2, g2  = self.attn2(l2_2, g)
                c3, g3  = self.attn3(l3_2, g)

            g       = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            output  = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            output = self.classify(torch.squeeze(g))

        return [output, c1, c2, c3]
