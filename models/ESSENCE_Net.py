import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from . import Transformer_block
from . import Transformer_block2
from einops import rearrange
from torch.nn import Linear


class Shading_Est(nn.Module):
    def __init__(self):
        super(Shading_Est, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(6, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = Transformer_block.Transformer(128)
        self.conv4 = Transformer_block.Transformer(128)
        self.conv5 = Transformer_block.Transformer(128)
        self.conv6 = Transformer_block.Transformer(128)
        self.conv7 = Transformer_block.Transformer(128)

        self.conv8 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.convs1 = Linear(128,64)
        self.convs2 = Linear(64,1)

        self.convs3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.convs4 = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        shading =  rearrange(out, 'b c n h w -> (b h w) n c')
        shading = self.convs1(shading)
        shading = self.convs2(shading)
        shading = shading.squeeze(2)

        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)

        intra_shading = out
        intra_shading = self.convs3(intra_shading)
        intra_shading = self.convs4(intra_shading)
        intra_shading = intra_shading.squeeze(1)

        out, _ = out.max(2)

        return out, shading,intra_shading


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.deconv1 = Transformer_block2.Transformer(128)
        self.deconv2 = Transformer_block2.Transformer(128)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, shadings_out):
        out = self.deconv1(shadings_out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class ESSENCENET(nn.Module):
    def __init__(self):
        super(ESSENCENET, self).__init__()
        self.regressor = Regressor()
        self.shading_est = Shading_Est()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        light = x[1]
        light_split = torch.split(light, 3, 1)

        input_shadings = []
        for i in range(len(img_split)):
            shading_net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            input_shadings.append(shading_net_in)
        input_shadings = torch.stack(input_shadings, 2)
        shadings_out, shading,intra_shading = self.shading_est(input_shadings)
        normal = self.regressor(shadings_out)
        return normal, shading,intra_shading

