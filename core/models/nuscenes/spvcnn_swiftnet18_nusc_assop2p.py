import torch
from torchsparse import PointTensor
# from torchsparse.point_tensor import PointTensor

from core.models.utils import *
from core.models.build_blocks import *
from core.models.fusion_blocks import Atten_Fusion_Conv, Feature_Gather, Feature_Fetch

from core.models.image_branch.swiftnet_nusc import SwiftNetRes18, _BNReluConv
# from core.models.image_branch.swiftnet_nusc import SwiftNetRes34, _BNReluConv
from torchpack.utils.config import configs

__all__ = ['SPVCNN_SWIFTNET_ASSOP2P']


class SPVCNN_SWIFTNET_ASSOP2P(nn.Module):

    def __init__(self, **kwargs):
        super(SPVCNN_SWIFTNET_ASSOP2P, self).__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        imagenet_pretrain = kwargs.get('imagenet_pretrain', None)
        self.pix_branch = SwiftNetRes18(num_feature=(128, 128, 128), pretrained_path=imagenet_pretrain)
        img_cs = self.pix_branch.img_cs

        self.in_channel = configs['model']['in_channel']
        self.num_classes = configs['data']['num_classes']
        self.out_channel = cs[-1]
        self.proj_channel = configs['model']['proj_channel']
        self.is_estep = configs['model']['is_estep']
        print('is_estep:', self.is_estep)

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_channel, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.vox_downs = nn.ModuleList()
        for idx in range(4):
            down = nn.Sequential(
                BasicConvolutionBlock(cs[idx], cs[idx], ks=2, stride=2, dilation=1),
                ResidualBlock(cs[idx], cs[idx + 1], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[idx + 1], cs[idx + 1], ks=3, stride=1, dilation=1)
            )
            self.vox_downs.append(down)

        self.fusion_blocks = nn.ModuleList()
        for idx in range(1, 5):
            fusion_block = Atten_Fusion_Conv(inplanes_I=img_cs[idx], inplanes_P=cs[idx], outplanes=cs[idx])
            self.fusion_blocks.append(fusion_block)
        # self.fusion_assop2p = Atten_Fusion_Conv(inplanes_I=self.proj_channel, inplanes_P=self.proj_channel, outplanes=self.proj_channel)
        # self.fusion_up1 = Atten_Fusion_Conv(inplanes_I=128, inplanes_P=cs[6], outplanes=cs[6])
        # self.fusion_up2 = Atten_Fusion_Conv(inplanes_I=128, inplanes_P=cs[8], outplanes=cs[8], return_att=True)

        self.vox_ups = nn.ModuleList()
        for idx in range(4, len(cs) - 1):
            up = nn.ModuleList([
                BasicDeconvolutionBlock(cs[idx], cs[idx + 1], ks=2, stride=2),
                nn.Sequential(
                    ResidualBlock(cs[idx + 1] + cs[len(cs) - 1 - (1 + idx)], cs[idx + 1], ks=3, stride=1, dilation=1),
                    ResidualBlock(cs[idx + 1], cs[idx + 1], ks=3, stride=1, dilation=1)
                )
            ])
            self.vox_ups.append(up)

        # self.projector2d = _BNReluConv(self.pix_branch.num_features, self.proj_channel, k=1)
        self.projector2d = nn.Sequential(
            _BNReluConv(self.pix_branch.num_features, self.proj_channel, k=1),
            nn.BatchNorm2d(self.proj_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.proj_channel, self.proj_channel, kernel_size=1)
        )
        self.projector3d = nn.Sequential(
            nn.Linear(cs[-1], self.proj_channel),
            nn.BatchNorm1d(self.proj_channel),
            nn.ReLU(True),
            nn.Linear(self.proj_channel, self.proj_channel)
        )

        self.classifier_vox = nn.Sequential(nn.Linear(cs[8], self.num_classes))
        self.classifier_pix = _BNReluConv(num_maps_in=self.pix_branch.num_features, num_maps_out=self.num_classes, k=1)

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, in_mod):
        """
        x: SparseTensor 表示voxel
        z: PointTensor 表示point
        Args:
            x: SparseTensor, x.C:(u,v,w,batch_idx), x.F:(x,y,z,sig)
        Returns:
        """
        x = in_mod['lidar']
        im = in_mod['images']  # [B, 6, H, W, C]
        # ib, ih, iw, ic = im.size(0), im.size(2), im.size(3), im.size(4)
        ib, _, ic, ih, iw = im.size()
        im = im.view(-1, ic, ih, iw)  # [B * 6, C, H, W]
        # im = im.view(-1, ic, ih, iw).contiguous()
        pixel_coordinates = in_mod['pixel_coordinates']
        masks = in_mod['masks']
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)  # x0.F(N, 4) -> x0.F(N, 20)
        z0 = voxel_to_point(x0, z, nearest=False)

        # im0 = self.pix_stem(im)
        img_feats = self.pix_branch.forward_down(im)

        vox_feats = [point_to_voxel(x0, z0)]
        for idx, (vox_block, fusion_block) in enumerate(zip(self.vox_downs, self.fusion_blocks)):

            vox_feat = vox_block(vox_feats[idx])
            pts_feat = voxel_to_point(vox_feat, z0)

            img_feat = img_feats[idx]
            img_feat = img_feat.view(ib, -1, img_feat.size(1), img_feat.size(2), img_feat.size(3))

            img_feat_tensor = []
            for mask, coord, img in zip(masks, pixel_coordinates, img_feat):
                """
                    mask <Tensor, [6, N]>
                    coord <Tensor, [6, N, 2]>
                    img <Tensor, [6, C, H, W]>
                """
                imf = torch.zeros(size=(mask.size(1), img.size(1))).cuda()
                imf_list = Feature_Gather(img, coord).permute(0, 2, 1)  # [B, N, C]
                assert mask.size(0) == coord.size(0)
                for idx in range(mask.size(0)):
                    imf[mask[idx]] = imf_list[idx, mask[idx], :]
                img_feat_tensor.append(imf)
            img_feat_tensor = torch.cat(img_feat_tensor, dim=0)

            pts_feat.F = fusion_block(pts_feat.F, img_feat_tensor.detach())  # conv+bn+relu->pts_feat; conv+bn->img_feat

            vox_feats.append(point_to_voxel(vox_feat, pts_feat))

        x1 = vox_feats[1]
        x2 = vox_feats[2]
        x3 = vox_feats[3]
        x4 = vox_feats[4]
        z1 = pts_feat

        z1.F = z1.F + self.point_transforms[0](z0.F)

        pix_upsamples = self.pix_branch.forward_up(img_feats)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.vox_ups[0][0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.vox_ups[0][1](y1)

        y2 = self.vox_ups[1][0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.vox_ups[1][1](y2)
        z2 = voxel_to_point(y2, z1)

        # pix_map = pix_upsamples[0]
        # pix_map = pix_map.view(ib, -1, pix_map.size(1), pix_map.size(2), pix_map.size(3))
        # pix_map = Feature_Fetch(masks, pixel_coordinates, pix_map)
        z2.F = z2.F + self.point_transforms[1](z1.F)
        # z2.F = self.fusion_up1(z2.F, pix_map.detach())

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.vox_ups[2][0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.vox_ups[2][1](y3)

        y4 = self.vox_ups[3][0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.vox_ups[3][1](y4)
        z3 = voxel_to_point(y4, z2)

        # pix_map = pix_upsamples[1]
        # pix_map = pix_map.view(ib, -1, pix_map.size(1), pix_map.size(2), pix_map.size(3))
        # pix_map = Feature_Fetch(masks, pixel_coordinates, pix_map)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        # z3.F, att = self.fusion_up2(z3.F, pix_map.detach())
        proj_embed3d = self.projector3d(z3.F)
        proj_embed2d = self.projector2d(pix_upsamples)

        vox_out = self.classifier_vox(z3.F)
        fmap_pix = self.classifier_pix(pix_upsamples)
        fmap_pix = fmap_pix.view(ib, -1, fmap_pix.size(1), fmap_pix.size(2), fmap_pix.size(3))
        pix_out = Feature_Fetch(masks, pixel_coordinates, fmap_pix)
        proj_embed2d = proj_embed2d.view(ib, -1, proj_embed2d.size(1), proj_embed2d.size(2), proj_embed2d.size(3))
        proj_embed2d = Feature_Fetch(masks, pixel_coordinates, proj_embed2d)
        # proj_embed2d = self.fusion_assop2p(proj_embed2d, proj_embed3d)

        return {
            'x_vox': vox_out,
            'embedding_vox': proj_embed3d if not self.is_estep else z3.F,
            'embedding_pix': proj_embed2d,
            # 'fmap_pix': fmap_pix,
            # 'embedding_pix': F.normalize(proj_embed2d, dim=-1),
            'x_pix': pix_out,
            # 'att': att.permute(2, 0, 1).contiguous().view(-1, 1),
            'num_pts': [coord.size(1) for coord in in_mod['pixel_coordinates']]
        }
