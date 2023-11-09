import numpy as np
import multiprocessing as mp
from itertools import chain
from scipy.optimize import linear_sum_assignment

import torch
#from torch_geometric.utils import to_dense_adj
from pytorch_fid.fid_score import calculate_frechet_distance

from training.networks_layoutnet import LayoutNet
from util import convert_xywh_to_ltrb
#from data.util import RelSize, RelLoc, detect_size_relation, detect_loc_relation


class LayoutFID():
    def __init__(self, pth, device='cpu'):
        num_label = 13 if 'rico' in pth or 'enrico' in pth or 'clay' in pth or 'ads_banner_collection' in pth else 5
        self.model = LayoutNet(num_label).to(device)

        # load pre-trained LayoutNet
        state_dict = torch.load(pth, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return lib.nan_to_num(iou)


def compute_iou_for_layout(layout_1, layout_2):
    (bi, li), (bj, lj) = layout_1, layout_2
    return compute_iou(bi, bj).mean().item()


def compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2, n_jobs=None):
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    with mp.Pool(n_jobs) as p:
        scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    return scores.mean().item()


def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0) # BN4
    bbox = bbox.permute(2, 0, 1) # 4BN

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1)) # BN1
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2)) # B1N
    a1 = (r1 - l1) * (b1 - t1) # BN1

    # intersection
    l_max = torch.maximum(l1, l2) # BNN
    r_min = torch.minimum(r1, r2) # BNN
    t_max = torch.maximum(t1, t2) # BNN
    b_min = torch.minimum(b1, b2) # BNN
    cond = (l_max < r_min) & (t_max < b_min) # BNN
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0])) # BNN

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device) # NN
    ai = ai.masked_fill(diag_mask, 0) # BNN

    ar = torch.nan_to_num(ai / a1) # BNN

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1) # B


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1) # 4BN
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox) # BN
    xc, yc = bbox[0], bbox[1] # BN
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1) # B6N

    X = X.unsqueeze(-1) - X.unsqueeze(-2) # B6NN
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3) # BN6N
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values # BN
    X.masked_fill_(X.eq(1.), 0.) # BN

    X = -torch.log(1 - X) # BN

    return X.sum(-1) / mask.float().sum(-1) # B


def compute_docsim_weight(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    xc1, yc1, w1, h1 = box_1.T
    xc2, yc2, w2, h2 = box_2.T
    location_difference = ((xc1-xc2)**2 + (yc1-yc2)**2)**0.5
    shape_difference = lib.abs(w1-w2) + lib.abs(h1-h2)
    area_factor = lib.minimum(w1*h1, w2*h2)**0.5
    weight = area_factor * 2**(-location_difference - 2.0*shape_difference)

    return weight


def compute_docsim_for_layout(layout_1, layout_2):
    (bi, li), (bj, lj) = layout_1, layout_2
    return compute_docsim_weight(bi, bj).mean().item()


def compute_maximum_docsim_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        weight = compute_docsim_weight(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(weight, maximize=True)
        score += weight[ii, jj].sum().item()
    return score / N


def generalized_iou_loss(layout_1, layout_2):
    # layout_1: [M, 4]  layout_2: [M, 4]

    l1, t1, r1, b1 = convert_xywh_to_ltrb(layout_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(layout_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                   torch.zeros_like(a1))

    # IoU
    au = a1 + a2 - ai
    iou = ai / au

    # minimal convex hull
    l_min = torch.minimum(l1, l2)
    r_max = torch.maximum(r1, r2)
    t_min = torch.minimum(t1, t2)
    b_max = torch.maximum(b1, b2)
    ah = (r_max - l_min) * (b_max - t_min)

    # generalized IoU
    g_iou = iou - (ah - au) / ah

    return (1.0 - g_iou).mean()
    

'''
def compute_violation(bbox_flatten, data):
    device = data.x.device
    failures, valid = [], []

    _zip = zip(data.edge_attr, data.edge_index.t())
    for gt, (i, j) in _zip:
        failure, _valid = 0, 0
        b1, b2 = bbox_flatten[i], bbox_flatten[j]

        # size relation
        if ~gt & 1 << RelSize.UNKNOWN:
            pred = detect_size_relation(b1, b2)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        # loc relation
        if ~gt & 1 << RelLoc.UNKNOWN:
            canvas = data.y[i].eq(0)
            pred = detect_loc_relation(b1, b2, canvas)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        failures.append(failure)
        valid.append(_valid)

    failures = torch.as_tensor(failures).to(device)
    failures = to_dense_adj(data.edge_index, data.batch, failures)
    valid = torch.as_tensor(valid).to(device)
    valid = to_dense_adj(data.edge_index, data.batch, valid)

    return failures.sum((1, 2)) / valid.sum((1, 2))
'''