import numpy as np
import torch
import torchvision.transforms.functional
from torch import nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

__all__ = ['Lovasz_softmax', 'MixLovaszCrossEntropy']


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() != 2:
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', ignore_index=0):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, classes=self.classes, ignore=self.ignore_index)


class MixLCLovaszCrossEntropy(nn.Module):
    def __init__(self, weight=None, classes='present', ignore_index=0):
        super(MixLCLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes, ignore_index=ignore_index)
        if weight is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            assert isinstance(weight, torch.Tensor)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, out_dict: dict, y):
        x_vox = out_dict.get('x_vox')
        x_pix = out_dict.get('x_pix')
        ce_vox = self.ce(x_vox, y)
        lovz_vox = self.lovasz(F.softmax(x_vox, 1), y)
        ce_pix = self.ce(x_pix, y)
        lovz_pix = self.lovasz(F.softmax(x_pix, 1), y)
        return {
            'predict_vox': ce_vox,
            'lovz_vox': lovz_vox,
            'predict_pix': ce_pix,
            'lovz_pix': lovz_pix
        }

class MixLovaszCrossEntropy(nn.Module):
    def __init__(self, weight=None, classes='present', ignore_index=255):
        super(MixLovaszCrossEntropy, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.lovasz = Lovasz_softmax(classes, ignore_index=ignore_index)
        if weight is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            assert isinstance(weight, torch.Tensor)
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, x, y):
        lovasz = self.lovasz(F.softmax(x, 1), y)
        ce = self.ce(x, y)
        return lovasz + ce

class SparseLCCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.3):

        super(SparseLCCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
        }

class SparseCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.3):

        super(SparseCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())

        return {
            'sp_vox': sp_vox,
            'pp_vox': pp_vox,
            'neg_vox': neg_vox,
        }

class SparseAssoP2PCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.1):

        super(SparseAssoP2PCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def _compute_assop2p_loss(self, x_dict: dict, target: torch.Tensor):

        def _compute_sim_mat(x, ref):
            # normalized_x = F.normalize(x, dim=-1)  # N_x, C
            # normalized_ref = F.normalize(ref, dim=-1)  # N_r, C
            ref = ref.permute(1, 0).contiguous()  # C, N_r
            sim_mat = x @ ref / 0.5 # N_x, N_r
            return sim_mat

        def _scoring(x, dim=-1):
            # eps = 1e-10
            # mean = torch.mean(x, dim=dim, keepdim=True).detach()
            # std = torch.std(x, dim=dim, keepdim=True).detach()
            # x = (x - mean) / (std + eps)
            score = F.softmax(x, dim=dim)
            return score

        def _build_correlation(x1, x2, metric='cos'):
            if metric == 'cos':
                sim_mat_12 = _compute_sim_mat(x1, x2)  # N_l, N_sp
                sim_mat_21 = sim_mat_12.transpose(0, 1).contiguous()  # N_sp, N_l
            else:
                raise NotImplementedError

            sim_mat_12 = _scoring(sim_mat_12)
            sim_mat_21 = _scoring(sim_mat_21)
            return sim_mat_12, sim_mat_21

        e3d = x_dict.get('embedding_vox')
        e2d = x_dict.get('embedding_pix')
        sp_target = torch.cat(x_dict.get('asso_target'), dim=0)
        m3d = target != self.ignore_index
        m2d = sp_target != self.ignore_index
        e3d = e3d[m3d]
        e2d = e2d[m2d]
        target = target[m3d]
        sim_mat_vp, sim_mat_pv = _build_correlation(e3d, e2d)
        sim_mat_vpv = sim_mat_vp @ sim_mat_pv
        asso_target = torch.eq(target.view(-1, 1), target).float()
        asso_target = asso_target / asso_target.sum(dim=-1, keepdim=True)
        asso_loss = self.kld(torch.log(sim_mat_vpv + 1e-10), asso_target.detach())  # calculate walker loss
        p_visit = torch.mean(sim_mat_vp, dim=0, keepdim=True)
        p_visit_target = torch.ones([1, p_visit.size()[1]], device=sim_mat_vp.device) / float(p_visit.size()[1])
        visit_loss = self.kld(torch.log(p_visit + 1e-10), p_visit_target.detach())  # calculate visit loss
        return asso_loss, visit_loss

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        asso_loss, vis_loss = self._compute_assop2p_loss(x_dict, y_sp)

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
            'asso_loss': asso_loss,
            'vis_loss': vis_loss
        }

class WaymoSparseAssoP2PCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.1):

        super(WaymoSparseAssoP2PCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def _compute_assop2p_loss(self, x_dict: dict, target: torch.Tensor):

        def _compute_sim_mat(x, ref):
            ref = ref.permute(1, 0).contiguous()  # C, N_r
            sim_mat = x @ ref  # N_x, N_r
            return sim_mat

        def _scoring(x, dim=-1):
            score = F.softmax(x, dim=dim)
            return score

        def _build_correlation(x1, x2, metric='cos'):
            if metric == 'cos':
                sim_mat_12 = _compute_sim_mat(x1, x2)  # N_l, N_sp
                sim_mat_21 = sim_mat_12.transpose(0, 1).contiguous()  # N_sp, N_l
            else:
                raise NotImplementedError

            sim_mat_12 = _scoring(sim_mat_12)
            sim_mat_21 = _scoring(sim_mat_21)
            return sim_mat_12, sim_mat_21

        e3d = x_dict.get('embedding_vox')
        e2d = x_dict.get('embedding_pix')
        asso_source = torch.cat(x_dict.get('asso_source'), dim=0)
        asso_target = torch.cat(x_dict.get('asso_target'), dim=0)
        m3d = asso_source != self.ignore_index
        m2d = asso_target != self.ignore_index
        e3d = e3d[m3d]
        e2d = e2d[m2d]
        asso_source = asso_source[m3d]
        sim_mat_vp, sim_mat_pv = _build_correlation(e3d, e2d)
        sim_mat_vpv = sim_mat_vp @ sim_mat_pv
        asso_mat = torch.eq(asso_source.view(-1, 1), asso_source).float()
        asso_mat = asso_mat / asso_mat.sum(dim=-1, keepdim=True)
        walker_loss = self.kld(torch.log(sim_mat_vpv + 1e-10), asso_mat.detach())
        p_visit = torch.mean(sim_mat_vp, dim=0, keepdim=True)
        p_visit_target = torch.ones([1, p_visit.size()[1]], device=sim_mat_vp.device) / float(p_visit.size()[1])
        visit_loss = self.kld(torch.log(p_visit + 1e-10), p_visit_target.detach())
        return walker_loss, visit_loss

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        asso_loss, vis_loss = self._compute_assop2p_loss(x_dict, y_sp)

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
            'asso_loss': asso_loss,
            'vis_loss': vis_loss
        }


class WaymoSparseAssoP2PPLCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.1):

        super(WaymoSparseAssoP2PPLCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def _compute_assop2p_loss(self, x_dict: dict, target: torch.Tensor):

        def _compute_sim_mat(x, ref):
            ref = ref.permute(1, 0).contiguous()  # C, N_r
            sim_mat = x @ ref  # N_x, N_r
            return sim_mat

        def _scoring(x, dim=-1):
            score = F.softmax(x, dim=dim)
            return score

        def _build_correlation(x1, x2, metric='cos'):
            if metric == 'cos':
                sim_mat_12 = _compute_sim_mat(x1, x2)  # N_l, N_sp
                sim_mat_21 = sim_mat_12.transpose(0, 1).contiguous()  # N_sp, N_l
            else:
                raise NotImplementedError

            sim_mat_12 = _scoring(sim_mat_12)
            sim_mat_21 = _scoring(sim_mat_21)
            return sim_mat_12, sim_mat_21

        e3d = x_dict.get('embedding_vox')
        e2d = x_dict.get('embedding_pix')
        asso_source = torch.cat(x_dict.get('asso_source'), dim=0)
        asso_target = torch.cat(x_dict.get('asso_target'), dim=0)
        m3d = asso_source != self.ignore_index
        m2d = asso_target != self.ignore_index
        e3d = e3d[m3d]
        e2d = e2d[m2d]
        asso_source = asso_source[m3d]
        sim_mat_vp, sim_mat_pv = _build_correlation(e3d, e2d)
        sim_mat_vpv = sim_mat_vp @ sim_mat_pv
        asso_mat = torch.eq(asso_source.view(-1, 1), asso_source).float()
        asso_mat = asso_mat / asso_mat.sum(dim=-1, keepdim=True)
        walker_loss = self.kld(torch.log(sim_mat_vpv + 1e-10), asso_mat.detach())
        p_visit = torch.mean(sim_mat_vp, dim=0, keepdim=True)
        p_visit_target = torch.ones([1, p_visit.size()[1]], device=sim_mat_vp.device) / float(p_visit.size()[1])
        visit_loss = self.kld(torch.log(p_visit + 1e-10), p_visit_target.detach())
        return walker_loss, visit_loss

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        asso_loss, vis_loss = self._compute_assop2p_loss(x_dict, y_sp)

        y_pl = y_dict.get('pseudo_label')

        pl_vox = self.lovasz_ce(x_vox, y_pl)
        pl_pix = self.lovasz_ce(x_pix, y_pl)

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
            'asso_loss': asso_loss,
            'vis_loss': vis_loss,
            'pl_vox': pl_vox,
            'pl_pix': pl_pix
        }

class SparsePLCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.3):

        super(SparsePLCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.temperature
        return self.lovasz_ce.ce(v, y)

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        y_pl = y_dict.get('pseudo_label')

        pl_vox = self.lovasz_ce(x_vox, y_pl)
        pl_pix = self.lovasz_ce(x_pix, y_pl)

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
            'pl_vox': pl_vox,
            'pl_pix': pl_pix
        }


class SparseAssoP2PPLCrossEntropy(nn.Module):

    def __init__(self, ignore_index=0, temperature=0.1):

        super(SparseAssoP2PPLCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.tau = temperature
        self.lovasz_ce = MixLovaszCrossEntropy(ignore_index=ignore_index)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def _compute_negative_loss(self, x, y):
        v_m = torch.sum(y.detach(), dim=-1)
        v_m = (v_m != 0)
        x = F.softmax(x[v_m], dim=-1)
        y = 1 - y[v_m]
        y[:, self.ignore_index] = 0
        neg = -torch.log(torch.clamp(1 - torch.sum(x * y, dim=-1), min=1e-7))
        return torch.mean(neg)

    def _compute_proto_loss(self, q, k, y):
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).transpose(1, 0).contiguous()
        v = torch.matmul(q, k.detach()) / self.tau
        return self.lovasz_ce.ce(v, y)

    def _compute_assop2p_loss(self, x_dict: dict, target: torch.Tensor):

        def _compute_sim_mat(x, ref):
            # normalized_x = F.normalize(x, dim=-1)  # N_x, C
            # normalized_ref = F.normalize(ref, dim=-1)  # N_r, C
            ref = ref.permute(1, 0).contiguous()  # C, N_r
            sim_mat = x @ ref  # N_x, N_r
            return sim_mat

        def _scoring(x, dim=-1):
            # eps = 1e-10
            # mean = torch.mean(x, dim=dim, keepdim=True).detach()
            # std = torch.std(x, dim=dim, keepdim=True).detach()
            # x = (x - mean) / (std + eps)
            score = F.softmax(x, dim=dim)
            return score

        def _build_correlation(x1, x2, metric='cos'):
            if metric == 'cos':
                sim_mat_12 = _compute_sim_mat(x1, x2)  # N_l, N_sp
                sim_mat_21 = sim_mat_12.transpose(0, 1).contiguous()  # N_sp, N_l
            else:
                raise NotImplementedError

            sim_mat_12 = _scoring(sim_mat_12)
            sim_mat_21 = _scoring(sim_mat_21)
            return sim_mat_12, sim_mat_21

        e3d = x_dict.get('embedding_vox')
        e2d = x_dict.get('embedding_pix')
        sp_target = torch.cat(x_dict.get('asso_target'), dim=0)
        m3d = target != self.ignore_index
        m2d = sp_target != self.ignore_index
        e3d = e3d[m3d]
        e2d = e2d[m2d]
        target = target[m3d]
        sim_mat_vp, sim_mat_pv = _build_correlation(e3d, e2d)
        sim_mat_vpv = sim_mat_vp @ sim_mat_pv
        asso_target = torch.eq(target.view(-1, 1), target).float()
        asso_target = asso_target / asso_target.sum(dim=-1, keepdim=True)
        asso_loss = self.kld(torch.log(sim_mat_vpv + 1e-10), asso_target.detach())  # calculate walker loss
        p_visit = torch.mean(sim_mat_vp, dim=0, keepdim=True)
        p_visit_target = torch.ones([1, p_visit.size()[1]], device=sim_mat_vp.device) / float(p_visit.size()[1])
        visit_loss = self.kld(torch.log(p_visit + 1e-10), p_visit_target.detach())  # calculate visit loss
        return asso_loss, visit_loss

    def forward(self, x_dict: dict, y_dict: dict):
        x_vox = x_dict.get('x_vox')
        x_pix = x_dict.get('x_pix')

        y_sp = y_dict.get('sparse_label')
        y_pp = y_dict.get('prop_label')
        y_neg = y_dict.get('neg_label')

        sp_vox = self.lovasz_ce(x_vox, y_sp)
        sp_pix = self.lovasz_ce(x_pix, y_sp)

        pp_vox = self.lovasz_ce(x_vox, y_pp)
        pp_pix = self.lovasz_ce(x_pix, y_pp)

        neg_vox = self._compute_negative_loss(x_vox, y_neg.detach())
        neg_pix = self._compute_negative_loss(x_pix, y_neg.detach())

        asso_loss, vis_loss = self._compute_assop2p_loss(x_dict, y_sp)

        y_pl = y_dict.get('pseudo_label')

        pl_vox = self.lovasz_ce(x_vox, y_pl)
        pl_pix = self.lovasz_ce(x_pix, y_pl)

        return {
            'sp_vox': sp_vox,
            'sp_pix': sp_pix,
            'pp_vox': pp_vox,
            'pp_pix': pp_pix,
            'neg_vox': neg_vox,
            'neg_pix': neg_pix,
            'asso_loss': asso_loss,
            'vis_loss': vis_loss,
            'pl_vox': pl_vox,
            'pl_pix': pl_pix
        }
