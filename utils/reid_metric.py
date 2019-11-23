# encoding: utf-8

import numpy as np
import torch
from ignite.metrics import Metric
from data.datasets.eval_reid import evaluate_all


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
    qg_normdot = qf_norm.mm(gf_norm.t())
    dist_mat = dist_mat.mul(1/qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1+epsilon,1-epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def euclidean_distance(qf,gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m,n) +\
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n,m).t()
    dist_mat.addmm_(1,-2,qf,gf.t())
    return dist_mat.cpu().numpy()


class EvalMetric(Metric):
    def __init__(self, num_query, data_name, max_rank=50, dis_method='cosine'):
        super(EvalMetric, self).__init__()
        self.num_query = num_query
        self.data_name = data_name
        self.max_rank = max_rank
        self.dis_method = dis_method

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.dis_method is 'euclidean':
            print("inference using Euclidean distance")
            distmat = euclidean_distance(qf, gf)
        else:
            print("inference using cosine distance")
            distmat = cosine_similarity(qf, gf)

        cmc = evaluate_all(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc

