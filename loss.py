from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch as th


class FCE_loss(th.nn.Module):
    def __init__(self):
        super(FCE_loss, self).__init__()

    def forward(self, feature1, feature2, margin):

        feature1 = feature1.float()
        feature2 = feature2.float()

        sim_matrix = th.matmul(feature1, feature2.t())


        sim_matrix = F.normalize(sim_matrix)

        dg = th.diag(sim_matrix)
        pos_score = th.exp(dg/margin)

        all_score1 = th.sum(sim_matrix, dim = 0)
        all_score1 = th.exp(all_score1/margin)

        all_score2 = th.sum(sim_matrix, dim = 1)
        all_score2 = th.exp(all_score2/margin)

        contrastive_loss = (-th.log(pos_score / all_score1) - th.log(pos_score / all_score2)).mean()
        return contrastive_loss

