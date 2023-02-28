import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.STN import SpatialTransformer, AffineTransformer
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)
class DoubleConvK1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels//8, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_base(nn.Module):
    def __init__(self, n_channels, chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)):
        super(UNet_base, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])
        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - Z % 16) % 16
        diffY = (16 - Y % 16) % 16
        diffX = (16 - X % 16) % 16
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x5, x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]


class GVSL(nn.Module):
    def __init__(self, n_channels=1, chan=(32, 64, 128, 256, 512, 256, 128, 64, 32), win=3):
        super(GVSL, self).__init__()
        self.unet = UNet_base(n_channels=n_channels, chs=chan)
        self.f_conv = DoubleConv(1024, 256)
        self.sp_conv = DoubleConv(64, 16)

        self.res_conv = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1),
                                      nn.GroupNorm(16//4, 16),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv3d(16, 1, 1))

        self.out_flow = nn.Conv3d(16, 3, 3, padding=1)
        self.fc_rot = nn.Linear(256, 3)
        self.softmax = nn.Softmax(1)
        self.fc_scl = nn.Linear(256, 3)
        self.fc_trans = nn.Linear(256, 3)
        self.fc_shear = nn.Linear(256, 6)

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.atn = AffineTransformer()
        self.stn = SpatialTransformer()

        self.win = win

    def get_affine_mat(self, rot, scale, translate, shear):
        theta_x = rot[:, 0]
        theta_y = rot[:, 1]
        theta_z = rot[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = torch.FloatTensor([[1, 0, 0], [0, torch.cos(theta_x), -torch.sin(theta_x)],
                                       [0, torch.sin(theta_x), torch.cos(theta_x)]]).cuda()
        rot_mat_x = rot_mat_x[np.newaxis, :, :]
        rot_mat_y = torch.FloatTensor([[torch.cos(theta_y), 0, torch.sin(theta_y)], [0, 1, 0],
                                       [-torch.sin(theta_y), 0, torch.cos(theta_y)]]).cuda()
        rot_mat_y = rot_mat_y[np.newaxis, :, :]
        rot_mat_z = torch.FloatTensor(
            [[torch.cos(theta_z), -torch.sin(theta_z), 0], [torch.sin(theta_z), torch.cos(theta_z), 0],
             [0, 0, 1]]).cuda()
        rot_mat_z = rot_mat_z[np.newaxis, :, :]
        scale_mat = torch.FloatTensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]]).cuda()
        scale_mat = scale_mat[np.newaxis, :, :]
        shear_mat = torch.FloatTensor(
            [[1, torch.tan(shear_xy), torch.tan(shear_xz)], [torch.tan(shear_yx), 1, torch.tan(shear_yz)],
             [torch.tan(shear_zx), torch.tan(shear_zy), 1]]).cuda()
        trans = torch.FloatTensor([trans_x, trans_y, trans_z]).cuda()
        trans = trans[np.newaxis, :, np.newaxis]
        mat = torch.matmul(shear_mat,
                           torch.matmul(scale_mat, torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        mat = torch.cat([mat, trans], dim=-1)
        return mat

    def Affine(self, m, f):
        # Affine
        x = torch.cat([m, f], dim=1)
        x = self.f_conv(x)
        xcor = self.gap(x).flatten(start_dim=1, end_dim=4)
        rot = self.fc_rot(xcor)
        scl = self.fc_scl(xcor)
        trans = self.fc_trans(xcor)
        shear = self.fc_shear(xcor)

        rot = torch.clamp(rot, -1, 1) * (np.pi / 9)
        scl = torch.clamp(scl, -1, 1) * 0.25 + 1
        shear = torch.clamp(shear, -1, 1) * (np.pi / 18)

        mat = self.get_affine_mat(rot, scl, trans, shear)
        return mat

    def Spatial(self, m, f):
        x = torch.cat([m, f], dim=1)
        sp_cor = self.sp_conv(x)
        flow = self.out_flow(sp_cor)
        return flow

    def forward(self, A, B):
        fA_g, fA_l = self.unet(A)
        fB_g, fB_l = self.unet(B)

        # Affine
        aff_mat_BA = self.Affine(fB_g, fA_g)
        aff_fBA_l = self.atn(fB_l, aff_mat_BA)

        # defore
        flow_BA = self.Spatial(aff_fBA_l, fA_l)

        # registration
        warp_BA = self.stn(self.atn(B, aff_mat_BA), flow_BA)

        # restoration
        res_A = self.res_conv(fA_l)

        return res_A, warp_BA, aff_mat_BA, flow_BA


class UNet3D_GVSL(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)):
        super(UNet3D_GVSL, self).__init__()
        self.n_channels = n_channels

        self.unet_pre = GVSL()
        self.unet_pre.load_state_dict(
            torch.load(
                '{0}/{1}_epoch_{2}.pth'.format('pre_weight/', 'GVSL', str(1000))))
        self.unet = self.unet_pre.unet

        self.out_conv = nn.Conv3d(chs[-1], n_classes, kernel_size=3, padding=1)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        _, x = self.unet(x)
        out = self.out_conv(x)

        return self.softmax(out)

