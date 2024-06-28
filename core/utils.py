import numpy as np
import torch

from core.models.fusion_blocks import Feature_Gather
from matplotlib import pyplot as plt
from visualize_utils import IDX2COLOR_16


def calculate_class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, max_label: int):

    embeddings = embeddings.view(-1, embeddings.shape[-1])
    count = torch.bincount(labels, minlength=max_label)[1:]
    mask = (count != 0)
    prototypes = torch.zeros((max_label, embeddings.shape[-1]),
                             dtype=embeddings.dtype,
                             device=embeddings.device)
    labels = labels.view(-1, 1).expand(-1, embeddings.shape[-1])
    prototypes = prototypes.scatter_add_(0, labels, embeddings)[1:, :]

    prototypes[mask] = prototypes[mask] / count[mask].unsqueeze(-1)

    return prototypes, mask


def Pesudo_Label_Generate(masks, pix_coord, pix_feats, targets):
    """

    Args:
        masks: <List[Tensor], B, [6, Nb]>
        pix_coord: <List[Tensor], B, [6, Nb, 2]>
        pix_feats: <Tensor, [B*6, C, H, W]>
        targets: <Tensor, [N,]>

    Returns:

    """
    bs = len(masks)
    c, h, w = pix_feats.size(1), pix_feats.size(2), pix_feats.size(3)
    pix_feats = pix_feats.view(bs, -1, c, h, w)
    cur = 0
    dense_labels = []
    # debug
    t_list, cm_list = [], []
    for mask, coord, pix_feat in zip(masks, pix_coord, pix_feats):
        """
            mask: <Tensor, [6, Nb]>
            coord: <Tensor, [6, Nb, 2]>
            pix_feat: <Tensor, [6, C, H, W]>
        """
        target = targets[cur: cur + mask.size(1)]  # [Nb,]
        proto_list = Feature_Gather(pix_feat, coord).permute(0, 2, 1).contiguous()  # [6, Nb, C]
        coord[:, :, 0] = (coord[:, :, 0] + 1.0) / 2 * (w - 1.0)
        coord[:, :, 1] = (coord[:, :, 1] + 1.0) / 2 * (h - 1.0)
        coord = torch.floor(coord).int()
        for ma, co, pix, proto in zip(mask, coord, pix_feat, proto_list):
            """
                ma: <Tensor, [Nb,]>
                co: <Tensor, [Nb, 2]>  (w, h)
                pix: <Tensor, [C, H, W]>
                proto: <Tensor, [Nb, C]>
            """
            # 计算图中存在的每个类的原型点
            t, c_m, pro = target[ma], co[ma], proto[ma]
            t_list.append(t.cpu().numpy())
            cm_list.append(c_m.cpu().numpy())
            # return t_list, cm_list
            t_uni = t.unique()
            center = []
            for ti in t_uni:
                ti_mask = t == ti
                center.append(torch.mean(pro[ti_mask], dim=0, keepdim=True))
            center = torch.cat(center, dim=0)  # [Nc, C]
            # 对于每个原型点，计算最小类间距
            n = center.size(0)
            # Compute pairwise distance, replace by the official when merged
            dist = torch.pow(center, 2).sum(dim=1, keepdim=True).expand(n, n)  # x1^2+x2^2
            dist = dist + dist.t()  # (x1^2+y1^2)+(x2^2+y2^2)
            # beta * dist + alpha * (center @ center.t())
            dist.addmm_(center, center.t(), beta=1, alpha=-2)  # (x1-y1)^2+(x2-y2)^2
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            t_mask = t_uni.expand(n, n).eq(t_uni.expand(n, n).t())
            dist_an = torch.cat([dist[i][t_mask[i] == 0].min().unsqueeze(0) for i in range(n)])  # [Nc,]
            # 计算类原型点与其他像素的距离，并认为距离小于最小类间距的所有像素的伪标签为对应的距离最近的原型点的伪标签
            pix_l = pix.view(-1, 1, c).contiguous()
            nl = pix_l.size(0)
            center = center.expand(nl, n, c)
            dist_pix = torch.pow(center - pix_l, 2).sum(dim=-1).sqrt()  # nl, n
            mask_a = torch.argmin(dist_pix, dim=-1)  # [nl,]
            mask_b = torch.tensor([d[i] < dist_an[i] for d, i, in zip(dist_pix, mask_a)]).cuda()  # [nl, ]
            print("sum of mask_b:", torch.sum(mask_b))
            t_dense = t_uni[mask_a] * mask_b
            t_dense = t_dense.view(h, w)
            for (i, j), y in zip(c_m, t):
                t_dense[j][i] = y
            dense_labels.append(t_dense.byte())
        cur += mask.size(1)
    return torch.stack(dense_labels, dim=0), cm_list, t_list
